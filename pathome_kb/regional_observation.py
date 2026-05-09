"""
pathome_kb/regional_observation.py
==================================
Per-(crop, disease, state) VLM regional observation + variation analysis.

Replaces the old two-stage `regional_extraction.py` (text-grounded
per-state) and `regional_image_fill.py` (image-grounded enum fill) with
a single image-aware stage that:

1. Reads the canonical disease record from `final_registry.json`
2. Locates the cached Bugwood photo for the (crop, disease, state) tuple
3. Asks Claude (vision-capable, via `claude -p` with the Read tool) to:
   - describe what THIS image shows (severity, lesion morphology,
     affected organs, spread pattern)
   - compare against the canonical text and emit
     `variations_from_canonical` bullets where the image disagrees with
     or refines the canonical description

Output: `regional_observations.json` per crop, mapping
``profile_id -> state -> observation_record``. The adapter turns each
record into a ``RegionalObservation`` with image-grounded citations.
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

def _format_canonical_brief(canonical: dict, max_chars: int = 2000) -> str:
    """Render a compact canonical-disease summary for the prompt."""
    def _v(field):
        if not isinstance(field, dict):
            return ""
        v = field.get("value")
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if x)
        return str(v or "")

    vs = canonical.get("visual_symptoms") or {}
    parts = [
        f"Pathogen: {_v(canonical.get('pathogen_scientific_name'))}",
        f"Type: {_v(canonical.get('type_of_disease'))}",
        f"Affected parts: {_v(canonical.get('affected_parts'))}",
        f"Summary: {_v(vs.get('summary'))}",
        f"Diagnostic features: {_v(vs.get('diagnostic_features'))}",
        f"Look-alikes: {_v(vs.get('look_alikes'))}",
    ]
    text = "\n".join(p for p in parts if p.split(": ", 1)[1].strip())
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[…canonical truncated]"
    return text


REGIONAL_OBSERVATION_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant
disease. Your task: describe what this specific image shows (NOT the
disease in general), then flag where the image diverges from the
canonical description.

Crop:    {crop}
Disease: {disease}
State:   {state}
Image:   {image_path}

CANONICAL REFERENCE (cross-region, from extension-service literature):
─────────────────────────────────────────────────────────────────────
{canonical_brief}
─────────────────────────────────────────────────────────────────────

Your job is to fill in this JSON object describing THIS photograph,
NOT the canonical text:

{{
  "severity":         "<one phrase: mild | moderate | advanced | late-season | early-onset | ...>",
  "severity_quote":   "<one short sentence describing what you see that justifies that severity>",

  "lesion_morphology":      "<one-sentence description of lesion shape, size, color, margin AS VISIBLE in this image>",
  "lesion_morphology_quote":"<short reinforcing quote of what you see>",

  "affected_organs":       ["<organ that THIS image clearly shows is affected>", "..."],
  "affected_organs_quote": "<short quote describing what you see on those organs>",

  "spread_pattern":        "<one phrase: lower canopy, scattered, uniform, edge of row, ...>",
  "spread_pattern_quote":  "<short quote describing the spread you see>",

  "variations_from_canonical": [
    "<bullet: how this image diverges from the canonical reference>",
    "<bullet: e.g. 'lesions appear ~3x larger than canonical 1/4-inch description'>",
    "<bullet: only include bullets that are SUPPORTED by the image>"
  ]
}}

Hard rules:
- Describe what is VISIBLE in the image. Do not transcribe the canonical
  reference. If a field cannot be determined from a single photo, set
  the value to an empty string and the quote to "".
- The variations_from_canonical list is for HONEST disagreements:
  the image shows worse/milder/earlier/later/different morphology than
  what the canonical text describes. If the image agrees with the
  canonical reference exactly, set this to an empty list.
"""

REGIONAL_OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "severity":              {"type": "string"},
        "severity_quote":        {"type": "string"},
        "lesion_morphology":     {"type": "string"},
        "lesion_morphology_quote": {"type": "string"},
        "affected_organs":       {"type": "array", "items": {"type": "string"}},
        "affected_organs_quote": {"type": "string"},
        "spread_pattern":        {"type": "string"},
        "spread_pattern_quote":  {"type": "string"},
        "variations_from_canonical": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "severity", "severity_quote",
        "lesion_morphology", "lesion_morphology_quote",
        "affected_organs", "affected_organs_quote",
        "spread_pattern", "spread_pattern_quote",
        "variations_from_canonical",
    ],
}


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
        save_json("regional_observations.json", {}, output_dir=output_dir)
        return {}

    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        futures = {pool.submit(_observe_one, t): t for t in todo}
        for fut in concurrent.futures.as_completed(futures):
            try:
                profile_id, state, record = fut.result()
                completed += 1
                n_variations = len(record.get("variations_from_canonical") or [])
                tag = "✓" if record.get("severity") else "·"
                print(f"  [{completed}/{len(todo)}] {tag} {profile_id} / {state}  "
                      f"variations={n_variations}")
                results[profile_id][state] = record
            except Exception as e:
                print(f"  ERROR: {e}")

    save_json("regional_observations.json", dict(results), output_dir=output_dir)
    print(f"  Saved: {output_dir / 'regional_observations.json'}")
    print(f"  Done in {time.time() - t0:.0f}s")
    return dict(results)
