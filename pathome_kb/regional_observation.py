"""
pathome_kb/regional_observation.py
==================================
Per-(crop, disease, state) regional delta extraction via the Qwen swarm.

For each (crop, disease, state) tuple with a cached Bugwood photo:

  1. Load the disease's canonical record + any existing regional
     deltas for THIS state from ``final_registry.json``.
  2. Run the 4-specialist + consolidator swarm in
     ``plantswarm.delta_pipeline`` (N stochastic traces + K-of-N
     agreement filter + conservative merge with existing).
  3. Persist the merged delta list back into ``final_registry.json``
     under ``diseases[*].regional_observations[state]``.
     States NOT processed this run are preserved verbatim.

The image cache is searched in (in order):
  - ``$PATHOME_IMAGE_CACHE_DIR`` if set
  - ``smoke/.bugwood_cache``
  - ``.bugwood_cache``
"""

from __future__ import annotations

import csv as _csv
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
# Cache lookup (env-overridable)
# ---------------------------------------------------------------------------

def _cache_dirs() -> List[Path]:
    dirs: List[Path] = []
    custom = os.environ.get("PATHOME_IMAGE_CACHE_DIR")
    if custom:
        dirs.append(Path(custom))
    dirs.append(Path("smoke/.bugwood_cache"))
    dirs.append(Path(".bugwood_cache"))
    return dirs


def _resolve_cached_image(image_id: str) -> Path | None:
    if not image_id.startswith("bugwood::"):
        return None
    number = image_id.split("::", 1)[1]
    for d in _cache_dirs():
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
    """Merge per-state records into final_registry.json.

    For each disease, the existing ``regional_observations`` dict is
    preserved as the starting point; only the states processed in this
    run are overwritten with the merged record (which itself already
    contains existing+new deltas, as produced by
    ``delta_pipeline.run_for_state``).

    Diseases not in ``regional_by_profile`` are left untouched.
    """
    crop = canonical_registry.get("crop", "")
    total_blocks = 0
    for d in canonical_registry.get("diseases", []) or []:
        name = (d.get("disease_name") or "").strip()
        profile_id = f"{crop}::{name}" if crop else name
        per_state = regional_by_profile.get(profile_id)
        if per_state is None:
            # Fall back to crop-less profile_id key just in case.
            for k, v in regional_by_profile.items():
                if k.endswith(f"::{name}"):
                    per_state = v
                    break
        existing = d.get("regional_observations") or {}
        if per_state:
            merged = dict(existing)
            merged.update(per_state)        # only this run's states replace
            d["regional_observations"] = merged
        elif not isinstance(existing, dict):
            d["regional_observations"] = {}
        total_blocks += len(d.get("regional_observations") or {})

    canonical_registry["regional_observations_count"] = total_blocks
    save_json("final_registry.json", canonical_registry, output_dir=output_dir)
    print(
        f"  Embedded {total_blocks} per-state observations into "
        f"{output_dir / 'final_registry.json'}"
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

    canonical_by_disease: Dict[str, dict] = {}
    for d in canonical_registry.get("diseases", []) or []:
        name = (d.get("disease_name") or "").strip()
        if name:
            canonical_by_disease[name] = d

    # Build the work list. For each (crop, disease, state), pull existing
    # regional deltas for THIS state so the swarm sees them as KB context
    # and merges conservatively on the way out.
    from plantswarm.delta_pipeline import WorkItem, existing_deltas_for_state

    todo: List[WorkItem] = []
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
        existing = existing_deltas_for_state(canonical, state)
        todo.append(WorkItem(
            profile_id=profile_id,
            crop=crop,
            disease=disease,
            state=state,
            image_path=image_path,
            image_ids=list(image_ids),
            canonical_record=canonical,
            primary_image_id=primary_image_id,
            existing_deltas=existing,
        ))

    if quick:
        per_disease: Dict[str, int] = defaultdict(int)
        capped: List[WorkItem] = []
        for w in todo:
            if per_disease[w.disease] >= 2:
                continue
            per_disease[w.disease] += 1
            capped.append(w)
        todo = capped

    print(f"  (profile, state) tuples to observe: {len(todo)}")
    if skipped_no_image:
        print(f"  [skipped: no cached image for {skipped_no_image} tuples]")
    if skipped_no_canonical:
        print(f"  [skipped: no canonical record for {skipped_no_canonical} tuples]")

    if not todo:
        # Nothing to do. Don't touch existing regional_observations.
        _embed_into_registry(canonical_registry, {}, output_dir)
        return {}

    from plantswarm.delta_pipeline import build_client_from_env, run_batch
    client = build_client_from_env()
    results = run_batch(todo, client=client, max_parallel=max_parallel)

    _embed_into_registry(canonical_registry, results, output_dir)
    print(f"  Done in {time.time() - t0:.0f}s")
    return results
