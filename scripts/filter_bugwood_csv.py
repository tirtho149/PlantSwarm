"""
scripts/filter_bugwood_csv.py
=============================
Emit a filtered ``BugWood_Diseases_usable.csv`` containing only the rows
that survive Pathome's normalisation + per-class threshold gate.

Default behaviour (paper-faithful, 10/class):
    python scripts/filter_bugwood_csv.py \
        --input  BugWood_Diseases.csv \
        --output BugWood_Diseases_usable.csv \
        --threshold 10

Adds five derived columns to each kept row so downstream consumers don't
need to redo the normalisation:

    NormCrop      — canonical crop label (Bugwood crop map)
    NormDisease   — common-name disease (parenthetical Latin stripped)
    StateLat,
    StateLon      — state-centroid coordinates (utils.geo)
    AezCode       — FAO AEZ code at those coordinates

A row is dropped when:
  - ``Host Name`` cannot be mapped to a crop (or maps to a non-crop key),
  - ``Subject Display Name`` is empty,
  - ``Location`` (US state) is empty or unrecognised,
  - the surviving (NormCrop, NormDisease) class has fewer than
    ``--threshold`` rows.

When ``--per-class N`` is set, each surviving class is additionally capped
to its first N rows (sorted by Image Number ascending) so the output
matches what ``BugwoodLoader`` will actually iterate.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bugwood_loader import _clean_disease, _map_crop  # noqa: E402
from utils.geo import aez_lookup, state_to_latlon  # noqa: E402


EXTRA_COLS = ["NormCrop", "NormDisease", "StateLat", "StateLon", "AezCode"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--input",  default="BugWood_Diseases.csv")
    p.add_argument("--output", default="BugWood_Diseases_usable.csv")
    p.add_argument("--threshold", type=int, default=10,
                   help="minimum rows per (crop, disease) class to keep")
    p.add_argument("--per-class", type=int, default=0,
                   help="optional cap on rows per class (0 = unlimited)")
    p.add_argument("--report", default=None,
                   help="optional path to write a per-class report TSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"input not found: {args.input}")

    # ------------------------------------------------------------------
    # Pass 1: normalise, drop blanks, group by (crop, disease)
    # ------------------------------------------------------------------
    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    total = len(rows)
    drop_no_crop = drop_no_disease = drop_no_state = 0
    enriched: List[dict] = []
    by_class: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx, row in enumerate(rows):
        crop = _map_crop(row.get("Host Name", ""))
        disease = _clean_disease(row.get("Subject Display Name", ""))
        state = (row.get("Location") or "").strip()
        if not crop:
            drop_no_crop += 1
            continue
        if not disease:
            drop_no_disease += 1
            continue
        lat, lon = state_to_latlon(state)
        if lat is None or lon is None:
            drop_no_state += 1
            continue
        aez = aez_lookup(lat, lon)
        row["NormCrop"] = crop
        row["NormDisease"] = disease
        row["StateLat"] = f"{lat:.6f}"
        row["StateLon"] = f"{lon:.6f}"
        row["AezCode"] = aez.code if aez else ""
        enriched.append(row)
        by_class[(crop, disease)].append(len(enriched) - 1)

    # ------------------------------------------------------------------
    # Pass 2: apply threshold + per-class cap, deterministic ordering
    # ------------------------------------------------------------------
    def _sort_key(eidx: int) -> int:
        n = (enriched[eidx].get("Image Number") or "").strip()
        return int(n) if n.isdigit() else 0

    kept_indices: List[int] = []
    classes_kept = 0
    classes_below_threshold = 0
    for cls, idxs in by_class.items():
        if len(idxs) < args.threshold:
            classes_below_threshold += 1
            continue
        idxs_sorted = sorted(idxs, key=_sort_key)
        if args.per_class and args.per_class > 0:
            idxs_sorted = idxs_sorted[: args.per_class]
        kept_indices.extend(idxs_sorted)
        classes_kept += 1

    kept_indices.sort()

    # ------------------------------------------------------------------
    # Write filtered CSV
    # ------------------------------------------------------------------
    out_fields = fieldnames + [c for c in EXTRA_COLS if c not in fieldnames]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for i in kept_indices:
            writer.writerow(enriched[i])

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"input:  {args.input} ({total} rows)")
    print(f"output: {args.output} ({len(kept_indices)} rows, {classes_kept} classes)")
    print()
    print("Drop reasons:")
    print(f"  no crop / non-crop host : {drop_no_crop}")
    print(f"  no disease label        : {drop_no_disease}")
    print(f"  no/unknown state        : {drop_no_state}")
    print(f"  class < {args.threshold} rows{' '*(20 - 13 - len(str(args.threshold)))}: "
          f"{sum(len(v) for k, v in by_class.items() if len(v) < args.threshold)} "
          f"rows across {classes_below_threshold} classes")
    if args.per_class:
        cap_dropped = sum(max(0, len(v) - args.per_class) for v in by_class.values()
                          if len(v) >= args.threshold)
        print(f"  per-class cap @ {args.per_class}      : {cap_dropped} rows trimmed")

    # ------------------------------------------------------------------
    # Optional per-class report
    # ------------------------------------------------------------------
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("crop\tdisease\tcandidate_rows\tkept_rows\n")
            for (crop, disease), idxs in sorted(by_class.items(),
                                                key=lambda kv: -len(kv[1])):
                if len(idxs) < args.threshold:
                    continue
                kept = min(len(idxs), args.per_class) if args.per_class else len(idxs)
                fh.write(f"{crop}\t{disease}\t{len(idxs)}\t{kept}\n")
        print(f"\nper-class report: {args.report}")


if __name__ == "__main__":
    main()
