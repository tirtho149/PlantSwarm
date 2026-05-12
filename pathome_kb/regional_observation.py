"""
pathome_kb/regional_observation.py
==================================
Per-(crop, disease, state) regional delta extraction via the Qwen swarm.

For each (crop, disease, state) tuple with a cached Bugwood photo:

  1. Load the disease's canonical record from ``final_registry.json``.
  2. Run the 4-specialist + consolidator swarm in
     ``plantswarm.delta_pipeline`` against (canonical KB, cached image).
  3. Persist the returned deltas back into ``final_registry.json`` under
     ``diseases[*].regional_observations[state]``.

Schema produced per (profile, state)::

    {
      "state":         "Alabama",
      "image_ids":     ["bugwood::1568038", ...],
      "deltas": [
        {
          "field":          "lesion_morphology",
          "canonical_says": "...",
          "image_shows":    "...",
          "image_quote":    "..."
        },
        ...
      ]
    }

The swarm requires a vLLM endpoint serving Qwen2.5-VL-7B (configured via
``VLLM_BASE_URL`` / ``VLLM_MODEL`` env vars — see
``plantswarm.delta_pipeline.build_client_from_env``).
"""

from __future__ import annotations

import csv as _csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import get_crop_dir, load_json, save_json


# ---------------------------------------------------------------------------
# CSV-driven (crop, disease, state) → image_id map
# ---------------------------------------------------------------------------

def build_state_image_map(
    csv_path: str | Path,
) -> Dict[Tuple[str, str, str], List[str]]:
    out: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            crop    = (row.get("NormCrop") or "").strip()
            disease = (row.get("NormDisease") or "").strip()
            state   = (row.get("Location") or "").strip()
            num     = (row.get("Image Number") or "").strip()
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
# Embed regional records into the unified final_registry.json
# ---------------------------------------------------------------------------

def _embed_into_registry(
    canonical_registry: dict,
    regional_by_profile: Dict[str, Dict[str, dict]],
    output_dir: Path,
) -> None:
    """Write per-state observations under each disease entry's
    ``regional_observations`` field and persist final_registry.json."""
    crop = canonical_registry.get("crop", "")
    for d in canonical_registry.get("diseases", []) or []:
        name = (d.get("disease_name") or "").strip()
        profile_id = f"{crop}::{name}" if crop else name
        per_state = regional_by_profile.get(profile_id) or {}
        if not per_state:
            # Fall back to crop-less profile_id key just in case.
            for k, v in regional_by_profile.items():
                if k.endswith(f"::{name}"):
                    per_state = v
                    break
        d["regional_observations"] = per_state

    canonical_registry["regional_observations_count"] = sum(
        len(v) for v in regional_by_profile.values()
    )
    save_json("final_registry.json", canonical_registry, output_dir=output_dir)
    print(
        f"  Embedded {canonical_registry['regional_observations_count']} "
        f"per-state observations into {output_dir / 'final_registry.json'}"
    )


# ---------------------------------------------------------------------------
# Per-crop runner
# ---------------------------------------------------------------------------

def run_regional_observation(
    crop: str,
    state_image_map: Dict[Tuple[str, str, str], List[str]],
    quick: bool = False,
    max_parallel: int = 4,
) -> Dict[str, Dict[str, dict]]:
    print(f"\n{'='*60}")
    print(f"REGIONAL OBSERVATION (Qwen swarm) — {crop}")
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

    # Build the work list.
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
        image_path: Path | None = None
        primary_image_id = ""
        for img_id in image_ids:
            p = _resolve_cached_image(img_id)
            if p:
                image_path = p
                primary_image_id = img_id
                break
        if image_path is None:
            skipped_no_image += 1
            continue
        profile_id = f"{crop}::{disease}"
        todo.append((
            profile_id, crop, disease, state,
            image_path, list(image_ids), canonical, primary_image_id,
        ))

    if quick:
        # Cap to 2 states per disease for fast iteration.
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
        # Embed an empty regional_observations field on every disease so the
        # registry has a stable shape even with no work to do.
        _embed_into_registry(canonical_registry, {}, output_dir)
        return {}

    # Hand off to the swarm orchestrator.
    from plantswarm.delta_pipeline import build_client_from_env, run_batch
    client = build_client_from_env()
    results = run_batch(todo, client=client, max_parallel=max_parallel)

    _embed_into_registry(canonical_registry, results, output_dir)
    print(f"  Done in {time.time() - t0:.0f}s")
    return results
