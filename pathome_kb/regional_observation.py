"""
pathome_kb/regional_observation.py
==================================
Per-(crop, disease, state) VLM regional observation + variation analysis.

Replaces the old two-stage `regional_extraction.py` (text-grounded
per-state) and `regional_image_fill.py` (image-grounded enum fill) with
a single image-aware stage that:

1. Reads the canonical disease record from `final_registry.json`
2. Locates the cached Bugwood photo for the (crop, disease, state) tuple
3. Asks Claude (vision-capable, via `claude -p` with the Read tool) to
   walk the canonical KB like a decision tree and emit ONLY deltas —
   structured ``{field, canonical_says, image_shows, image_quote}``
   records — that ADD to or CONTRADICT the canonical description for
   THIS state's image. If the image confirms canonical exactly, the
   list is empty. There is no parallel re-extraction of severity /
   morphology / etc; canonical is the source of truth for those.

Output: per-state observation records are embedded INTO the same
``final_registry.json`` under each disease entry's
``regional_observations`` field (one unified registry per crop, no
separate file). The adapter turns each record into a
``RegionalObservation`` with image-grounded citations.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .config import MAX_PARALLEL_EXTRACTIONS
from .shared import claude_query_with_image, parse_json_result
from .utils import get_crop_dir, load_json, save_json


# ---------------------------------------------------------------------------
# CSV-driven (crop, disease, state) → image_id map
# (kept here to avoid the dependency on the deleted regional_extraction.py)
# ---------------------------------------------------------------------------

import csv as _csv


def build_state_image_map(csv_path: str | Path) -> Dict[Tuple[str, str, str], List[str]]:
    out: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            crop = (row.get("NormCrop") or "").strip()
            disease = (row.get("NormDisease") or "").strip()
            state = (row.get("Location") or "").strip()
            num = (row.get("Image Number") or "").strip()
            if not (crop and disease and state and num):
                continue
            out[(crop, disease, state)].append(f"bugwood::{num}")
    return dict(out)


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------

_CACHE_DIRS = [
    Path("smoke/.bugwood_cache"),
    Path(".bugwood_cache"),
]


def _resolve_cached_image(image_id: str) -> Path | None:
    if not image_id.startswith("bugwood::"):
        return None
    number = image_id.split("::", 1)[1]
    for d in _CACHE_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = d / f"{number}{ext}"
            if p.is_file() and p.stat().st_size > 0:
                return p.resolve()
    return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _format_canonical_brief(canonical: dict, max_chars: int = 2400) -> str:
    """Render canonical fields keyed by the SAME field-names the prompt
    asks the VLM to compare against, so the decision-tree mapping is
    explicit. Empty fields are rendered as '(not specified)' so the VLM
    knows which slots it can legitimately fill with image evidence.
    """
    def _v(field):
        if not isinstance(field, dict):
            return ""
        v = field.get("value")
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if x)
        return str(v or "")

    vs = canonical.get("visual_symptoms") or {}
    field_map = [
        ("pathogen",            _v(canonical.get("pathogen_scientific_name"))),
        ("type_of_disease",     _v(canonical.get("type_of_disease"))),
        ("affected_organs",     _v(canonical.get("affected_parts"))),
        ("lesion_morphology",   _v(vs.get("summary"))),
        ("diagnostic_features", _v(vs.get("diagnostic_features"))),
        ("look_alikes",         _v(vs.get("look_alikes"))),
        ("treatments",          _v(canonical.get("treatments"))),
    ]
    lines = [
        f"  - {name}: {value if value.strip() else '(not specified)'}"
        for name, value in field_map
    ]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n  […canonical truncated]"
    return text


REGIONAL_OBSERVATION_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant
disease. The CANONICAL knowledge base for this disease is given below
— it is already the source of truth for symptoms, morphology, severity
typical of this disease, look-alikes, and treatments.

Your job is NOT to re-describe the disease. Your job is to walk the
canonical KB like a decision tree and emit ONLY observations from THIS
photograph that ADD to or CONTRADICT the canonical text — the
state-specific delta. If the image confirms canonical exactly, return
an empty list.

Crop:    {crop}
Disease: {disease}
State:   {state}
Image:   {image_path}

CANONICAL KB (already populated — DO NOT repeat its contents):
─────────────────────────────────────────────────────────────────────
{canonical_brief}
─────────────────────────────────────────────────────────────────────

Allowed delta fields (pick the one your observation refines or
contradicts; use "other" only if none fit):
  - lesion_morphology    (size / shape / margin / color of lesions)
  - severity             (advancement / extent at this site)
  - affected_organs      (which plant parts THIS image shows affected)
  - spread_pattern       (canopy distribution: lower / upper / scattered / uniform)
  - diagnostic_features  (a sign visible here that's not in canonical's list)
  - look_alikes          (something that COULD be confused with what's shown)
  - treatments           (rare — only if image evidence implies a treatment-relevant
                          stage that canonical doesn't already cover)
  - other

Output JSON:
{{
  "deltas": [
    {{
      "field":          "<one of the allowed fields above>",
      "canonical_says": "<short quote from CANONICAL KB above on this field, OR '(not specified)' if canonical is silent>",
      "image_shows":    "<what THIS image adds or contradicts — one sentence, state-specific>",
      "image_quote":    "<one-sentence visual evidence from the image — what you literally see>"
    }}
  ]
}}

Hard rules:
- Each delta MUST be supported by something visible in the image. If you
  cannot point to visual evidence, omit the delta.
- Do NOT emit a delta that just restates canonical text. Restating
  canonical is forbidden — that is the redundant pass we are removing.
- "(not specified)" is the right value for canonical_says when the
  canonical KB is silent on that field but the image shows something
  worth recording.
- If the image is in full agreement with canonical, return {{"deltas": []}}.
"""

REGIONAL_OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "deltas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field":          {"type": "string"},
                    "canonical_says": {"type": "string"},
                    "image_shows":    {"type": "string"},
                    "image_quote":    {"type": "string"},
                },
                "required": ["field", "canonical_says", "image_shows", "image_quote"],
            },
        },
    },
    "required": ["deltas"],
}


# ---------------------------------------------------------------------------
# Embed regional records into the unified final_registry.json
# ---------------------------------------------------------------------------

def _embed_into_registry(
    canonical_registry: dict,
    regional_by_profile: Dict[str, Dict[str, dict]],
    output_dir: Path,
) -> None:
    """Write the regional observations INTO final_registry.json.

    For each disease entry, attach a ``regional_observations`` dict keyed
    by state. The result is one self-contained registry per crop —
    canonical fields up top, per-state observations nested under each
    disease — instead of two files the caller has to merge.
    """
    crop = canonical_registry.get("crop", "")
    diseases = canonical_registry.get("diseases", []) or []
    for d in diseases:
        name = (d.get("disease_name") or "").strip()
        profile_id = f"{crop}::{name}" if crop else name
        # Try a couple of profile_id forms just in case the regional pass
        # used a different crop string in the key.
        per_state = regional_by_profile.get(profile_id) or {}
        if not per_state:
            for k, v in regional_by_profile.items():
                if k.endswith(f"::{name}"):
                    per_state = v
                    break
        d["regional_observations"] = per_state

    canonical_registry["regional_observations_count"] = sum(
        len(v) for v in regional_by_profile.values()
    )

    save_json("final_registry.json", canonical_registry, output_dir=output_dir)
    print(f"  Embedded {canonical_registry['regional_observations_count']} "
          f"per-state observations into {output_dir / 'final_registry.json'}")


# ---------------------------------------------------------------------------
# Per-call worker
# ---------------------------------------------------------------------------

def _observe_one(args: tuple) -> Tuple[str, str, dict]:
    profile_id, crop, disease, state, image_path, image_ids, canonical_brief = args
    prompt = REGIONAL_OBSERVATION_PROMPT.format(
        crop=crop, disease=disease, state=state,
        image_path=str(image_path),
        canonical_brief=canonical_brief,
    )
    raw = claude_query_with_image(
        prompt=prompt,
        image_path=image_path,
        system_prompt=(
            "You are a plant pathology vision agent. Describe only what is "
            "visible in the provided image; never recite generic literature. "
            "Output strictly JSON matching the schema."
        ),
        json_schema=REGIONAL_OBSERVATION_SCHEMA,
        max_turns=5,
        timeout_secs=240,
    )
    record = parse_json_result(raw, f"regional_obs_{profile_id}_{state}")
    if not isinstance(record, dict):
        record = {}
    record["__image_ids__"] = image_ids
    record["state"] = state
    return profile_id, state, record


# ---------------------------------------------------------------------------
# Per-crop runner
# ---------------------------------------------------------------------------

def run_regional_observation(
    crop: str,
    state_image_map: Dict[Tuple[str, str, str], List[str]],
    quick: bool = False,
) -> Dict[str, Dict[str, dict]]:
    print(f"\n{'='*60}")
    print(f"REGIONAL OBSERVATION — {crop}")
    print(f"{'='*60}")
    t0 = time.time()
    output_dir = get_crop_dir(crop)

    final_path = output_dir / "final_registry.json"
    if not final_path.is_file():
        print(f"  [skip] no final_registry.json for {crop} — run reconciliation first.")
        return {}
    canonical_registry = load_json("final_registry.json", output_dir=output_dir)

    # Map disease_name → canonical record
    canonical_by_disease: Dict[str, dict] = {}
    for d in canonical_registry.get("diseases", []) or []:
        name = (d.get("disease_name") or "").strip()
        if name:
            canonical_by_disease[name] = d

    # Build the per-(crop, disease, state) work list, filtered by image
    # availability and the canonical record being non-empty.
    todo: List[tuple] = []
    skipped_no_image = 0
    skipped_no_canonical = 0
    for (c, disease, state), image_ids in state_image_map.items():
        if c != crop:
            continue
        canonical = canonical_by_disease.get(disease)
        if not canonical:
            skipped_no_canonical += 1
            continue
        # Pick the first cached image for this state
        image_path = None
        for img_id in image_ids:
            p = _resolve_cached_image(img_id)
            if p:
                image_path = p
                break
        if image_path is None:
            skipped_no_image += 1
            continue
        canonical_brief = _format_canonical_brief(canonical)
        profile_id = f"{crop}::{disease}"
        todo.append((profile_id, crop, disease, state, image_path, image_ids, canonical_brief))

    if quick:
        # Cap to 2 states/disease for fast iteration
        per_disease: Dict[str, int] = defaultdict(int)
        capped: List[tuple] = []
        for t in todo:
            if per_disease[t[2]] >= 2:
                continue
            per_disease[t[2]] += 1
            capped.append(t)
        todo = capped

    print(f"  (profile, state) tuples to observe: {len(todo)}")
    if skipped_no_image:
        print(f"  [skipped: no cached image for {skipped_no_image} tuples]")
    if skipped_no_canonical:
        print(f"  [skipped: no canonical record for {skipped_no_canonical} tuples]")
    if not todo:
        # Even with no work, embed an empty regional_observations field on
        # every disease so the registry has a stable shape.
        _embed_into_registry(canonical_registry, {}, output_dir)
        return {}

    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        futures = {pool.submit(_observe_one, t): t for t in todo}
        for fut in concurrent.futures.as_completed(futures):
            try:
                profile_id, state, record = fut.result()
                completed += 1
                deltas = record.get("deltas") or []
                tag = "✓" if deltas else "·"
                print(f"  [{completed}/{len(todo)}] {tag} {profile_id} / {state}  "
                      f"deltas={len(deltas)}")
                results[profile_id][state] = record
            except Exception as e:
                print(f"  ERROR: {e}")

    # Embed the per-state observations under each matching disease entry in
    # final_registry.json so we have ONE unified registry per crop.
    _embed_into_registry(canonical_registry, dict(results), output_dir)
    print(f"  Done in {time.time() - t0:.0f}s")
    return dict(results)
