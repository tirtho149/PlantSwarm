"""
pathome_kb/pipeline.py
======================
Orchestrate the SAGE-style internet-track KB build for the 484 PathomeDB
classes.

Flow per crop:
    1. group BugWood_Diseases_usable.csv by NormCrop
    2. for each crop: run pathome_kb.internet_pipeline.run_internet_pipeline
       with that crop's NormDisease list as ``disease_names``
    3. merge per-crop registries via symptoms_adapter
    4. emit ``artifacts/pathome_seed/symptoms_seed.json`` consumable by
       ``pathome.SymptomLibrary.load`` (and therefore by Phase 1
       ``scripts/build_pathome.py``)

CLI:
    python -m pathome_kb \
        --csv BugWood_Diseases_usable.csv \
        --out artifacts/pathome_seed/symptoms_seed.json \
        [--quick] [--limit-crops N] [--resume-from STAGE] \
        [--only-crops "Tomato,Soybean,Corn"] \
        [--keep-cached]

Each per-crop run lands under ``artifacts/pathome_kb/<Crop>/`` with the
SAGE artefacts intact: discovery_results.json, raw_extractions.json,
final_registry.json, registry.md, internet.xlsx. Re-running per-crop is
idempotent (each stage writes its artefact and is skipped when present).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .internet_pipeline import run_internet_pipeline
from .symptoms_adapter import merge_registries_to_seed, write_seed_json
from .utils import OUTPUT_DIR, get_crop_dir, load_json


# ---------------------------------------------------------------------------
# CSV → per-crop disease lists
# ---------------------------------------------------------------------------

def load_classes_from_csv(csv_path: Path) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """Return ({crop: [disease, ...]}, [(crop, disease), ...]) from the usable CSV."""
    by_crop: Dict[str, set] = defaultdict(set)
    expected: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            crop = (row.get("NormCrop") or "").strip()
            disease = (row.get("NormDisease") or "").strip()
            if not crop or not disease:
                continue
            if disease not in by_crop[crop]:
                by_crop[crop].add(disease)
                expected.append((crop, disease))
    return ({c: sorted(d) for c, d in by_crop.items()}, expected)


# ---------------------------------------------------------------------------
# Per-crop runner
# ---------------------------------------------------------------------------

def _crop_registry_path(crop: str) -> Path:
    return get_crop_dir(crop) / "final_registry.json"


def run_one_crop(
    crop: str,
    diseases: List[str],
    quick: bool = False,
    resume_from: str | None = None,
    keep_cached: bool = True,
) -> dict | None:
    """Run the internet track for one crop. Returns the registry dict.

    If ``keep_cached`` is True and a final_registry.json already exists for
    this crop, we skip the run and return the cached registry. This is the
    default so the orchestrator is cheap to re-run after a partial failure.
    """
    final_path = _crop_registry_path(crop)
    if keep_cached and final_path.is_file():
        try:
            return load_json("final_registry.json", output_dir=get_crop_dir(crop))
        except Exception:
            pass  # fall through to fresh run

    print(f"\n##### {crop} ({len(diseases)} diseases) #####")
    try:
        return run_internet_pipeline(
            crop=crop,
            disease_names=diseases,
            quick=quick,
            resume_from=resume_from,
        )
    except Exception as e:
        print(f"  ERROR ({crop}): {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", default="BugWood_Diseases_usable.csv",
                   help="filtered Bugwood CSV to drive the disease list")
    p.add_argument("--out", default="artifacts/pathome_seed/symptoms_seed.json",
                   help="output seed JSON consumable by SymptomLibrary.load")
    p.add_argument("--quick", action="store_true",
                   help="quick mode (fewer sources, shorter timeouts) — for smoke tests")
    p.add_argument("--limit-crops", type=int, default=0,
                   help="run only the first N crops (alphabetical) — 0 = all")
    p.add_argument("--only-crops", default="",
                   help='comma-separated crop allowlist, e.g. "Tomato,Soybean,Corn"')
    p.add_argument("--resume-from", default=None,
                   choices=["discovery", "extraction", "reconciliation"],
                   help="resume each crop from this stage (use cached upstream artefacts)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached per-crop final_registry.json — re-run from scratch")
    p.add_argument("--seed-min-observations", type=int, default=3,
                   help="min_observations carried into the SymptomLibrary seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    by_crop, expected = load_classes_from_csv(csv_path)
    print(f"loaded {len(expected)} (crop, disease) pairs across {len(by_crop)} crops")

    crops = sorted(by_crop.keys())
    if args.only_crops:
        allow = {c.strip() for c in args.only_crops.split(",") if c.strip()}
        crops = [c for c in crops if c in allow]
        print(f"  --only-crops gate: {len(crops)} crops will run")
    if args.limit_crops > 0:
        crops = crops[: args.limit_crops]
        print(f"  --limit-crops gate: {len(crops)} crops will run")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_cached = not args.no_cache
    t0 = time.time()
    registries: List[Tuple[str, dict]] = []
    failures: List[str] = []

    for i, crop in enumerate(crops, 1):
        diseases = by_crop[crop]
        print(f"\n[{i}/{len(crops)}] {crop}")
        registry = run_one_crop(
            crop=crop,
            diseases=diseases,
            quick=args.quick,
            resume_from=args.resume_from,
            keep_cached=keep_cached,
        )
        if registry is None:
            failures.append(crop)
            continue
        registries.append((crop, registry))

    print(f"\nfinished {len(registries)}/{len(crops)} crops "
          f"in {time.time() - t0:.0f}s; failures: {len(failures)}")
    if failures:
        print("  failed crops: " + ", ".join(failures))
        print("  re-run pipeline.py to retry just the failed ones (cached crops skipped)")

    seed_payload = merge_registries_to_seed(
        registries=registries,
        expected_classes=expected,
        min_observations=args.seed_min_observations,
    )
    out_path = Path(args.out)
    write_seed_json(seed_payload, out_path)

    n_with_data = sum(
        1 for prof in seed_payload["profiles"]
        if (prof.get("visual") or {}).get("notes") or (prof.get("visual") or {}).get("distinctive_signs")
    )
    print(f"\nseed written: {out_path}")
    print(f"  profiles total      : {len(seed_payload['profiles'])}")
    print(f"  profiles with data  : {n_with_data}")
    print(f"  profiles still empty: {len(seed_payload['profiles']) - n_with_data}")


if __name__ == "__main__":
    main()
