"""
scripts/build_pathome.py
========================
Build PathomeDB v1 from a Bugwood-curated tree.

Reads ``configs/bugwood_pathome.yaml``, walks the 7-trace and 3-reference
splits, and serialises all five layers under ``pathome.out_dir``.

Usage:
    python scripts/build_pathome.py --config configs/bugwood_pathome.yaml
    python scripts/build_pathome.py --config configs/bugwood_pathome.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from data.bugwood_loader import BugwoodLoader
from pathome import PathomeDB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bugwood_pathome.yaml")
    p.add_argument("--out_dir", default=None,
                   help="override pathome.out_dir from config")
    p.add_argument("--dry-run", action="store_true",
                   help="print stats but do not write the database")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir or cfg["pathome"]["out_dir"]
    source = cfg["data"].get("csv_path") or cfg["data"].get("bugwood_root", "<unset>")
    print(f"Building PathomeDB → {out_dir}")
    print(f"Source: {source}")

    print("\nLoading Bugwood records...")
    trace_loader = BugwoodLoader(cfg["data"], split="trace")
    ref_loader = BugwoodLoader(cfg["data"], split="reference")

    trace_records = list(trace_loader)
    ref_records = list(ref_loader)
    print(f"  trace split: {len(trace_records)} images")
    print(f"  reference split: {len(ref_records)} images")

    if not trace_records:
        raise SystemExit("No Bugwood images found — check bugwood_root in config.")

    # Class breakdown
    classes = {}
    for r in trace_records:
        key = (r.crop_species, r.disease_name)
        classes.setdefault(key, 0)
        classes[key] += 1
    print(f"  classes: {len(classes)}; mean trace images/class = "
          f"{len(trace_records) / max(len(classes), 1):.1f}")

    # GPS coverage check (paper §5.1: GPS required)
    with_gps = sum(1 for r in trace_records if r.lat is not None and r.lon is not None)
    print(f"  GPS coverage: {with_gps}/{len(trace_records)} trace images "
          f"({100*with_gps/max(len(trace_records),1):.1f}%)")

    print("\nBuilding PathomeDB...")
    db = PathomeDB.build_from_bugwood(
        trace_records=trace_records,
        reference_records=ref_records,
        symptoms_path=cfg["pathome"].get("symptoms_path"),
        version=cfg["pathome"].get("version", "v2.0"),
    )

    populated_states = {
        s for prof in db.symptoms for s in prof.state_counts
    }
    profiles_with_visual = sum(
        1 for prof in db.symptoms if not prof.canonical.is_empty()
    )
    print(f"  symptoms : {len(db.symptoms)} (crop, disease) profiles "
          f"({profiles_with_visual} with curated visual descriptions)")
    print(f"  geo      : {len(populated_states)} states observed across "
          f"{sum(p.total_observations for p in db.symptoms)} records")
    print(f"  refs     : {len(db.refs)} held-out reference images")

    if args.dry_run:
        print("\n[dry-run] Skipping save.")
        return

    db.save(out_dir)
    print(f"\nPathomeDB v{db.version} saved to {out_dir}")

    summary = {
        "version": db.version,
        "trace_records": len(trace_records),
        "reference_records": len(ref_records),
        "classes": len(classes),
        "gps_coverage": with_gps,
        "symptom_profiles": len(db.symptoms),
        "profiles_with_visual": profiles_with_visual,
        "states_observed": len(populated_states),
        "refs_size": len(db.refs),
    }
    with open(os.path.join(out_dir, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
