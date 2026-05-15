"""
scripts/ensure_state_image_cache.py
====================================
Top up ``.bugwood_cache`` (or ``smoke/.bugwood_cache``) so it holds at
least one image per ``(crop, disease, state)`` tuple in the filtered
CSV. Phase 1's BugwoodLoader downloads ``per_class=N`` images grouped
by (crop, disease) only, which leaves states with fewer representative
images uncached — and the image-fill stage then has to skip those
tuples.

This script is a fast top-up: for every (crop, disease, state) tuple
in BugWood_Diseases_usable.csv (or your --csv override), pick the first
image (by Image Number ascending) and download it via the same
``urlopen`` path BugwoodLoader uses. Already-cached files are no-ops.

CLI:
    python scripts/ensure_state_image_cache.py \
        --csv smoke/BugWood_Diseases_smoke_usable.csv \
        --cache-dir smoke/.bugwood_cache

After running, smoke/run_phase0_image_fill.sh should report 0 skips.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fetch(url: str, dest: Path, timeout: float = 20.0, retries: int = 2) -> bool:
    """Download URL → dest. Idempotent. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "PlantSwarm/Pathome (research)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise IOError("empty response")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    try:
        dest.with_suffix(dest.suffix + ".failed").write_text(
            f"{url}\n{type(last_err).__name__}: {last_err}\n"
        )
    except Exception:
        pass
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--csv", default="smoke/BugWood_Diseases_smoke_usable.csv",
                   help="filtered Bugwood CSV with NormCrop/NormDisease/Location/Image URL")
    p.add_argument("--cache-dir", default="smoke/.bugwood_cache",
                   help="where downloaded JPEGs land")
    p.add_argument("--all-rows", action="store_true",
                   help="download EVERY row in the CSV (deduped by Image Number), "
                        "not just the lowest-numbered image per (crop, disease, state). "
                        "Recommended for Phase 0R so any image_id referenced by the "
                        "swarm resolves to a cached file.")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel downloads (default 1). Use 8-16 for full-CSV runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="just count how many fetches WOULD happen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Group by (crop, disease, state) → keep ALL rows so we can sort by Image Number
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crop = (row.get("NormCrop") or "").strip()
            disease = (row.get("NormDisease") or "").strip()
            state = (row.get("Location") or "").strip()
            url = (row.get("Image URL") or "").strip()
            num = (row.get("Image Number") or "").strip()
            if not (crop and disease and state and url and num):
                continue
            groups[(crop, disease, state)].append({"num": num, "url": url})

    print(f"loaded {sum(len(v) for v in groups.values())} rows across "
          f"{len(groups)} (crop, disease, state) tuples")

    # Build candidate list. In --all-rows mode every row is a candidate;
    # otherwise only the lowest Image Number per (crop, disease, state).
    # Dedupe by Image Number — the same image often appears across rows.
    seen_nums: set[str] = set()
    todo: List[Tuple[Tuple[str, str, str], str, Path]] = []
    already_cached = 0
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r["num"]) if r["num"].isdigit() else 0)
        candidates = rows if args.all_rows else rows[:1]
        for chosen in candidates:
            if chosen["num"] in seen_nums:
                continue
            seen_nums.add(chosen["num"])
            ext = os.path.splitext(chosen["url"].split("?", 1)[0])[1] or ".jpg"
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            dest = cache_dir / f"{chosen['num']}{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                already_cached += 1
                continue
            todo.append((key, chosen["url"], dest))

    print(f"mode:           {'ALL rows' if args.all_rows else 'first-per-tuple'}")
    print(f"already cached: {already_cached}")
    print(f"to download:    {len(todo)} (workers={args.workers})")

    if args.dry_run:
        for key, url, dest in todo[:10]:
            print(f"  [would fetch] {key[0]} / {key[1]} / {key[2]} → {dest.name}")
        if len(todo) > 10:
            print(f"  ... and {len(todo)-10} more")
        return

    n_ok = n_fail = 0
    t0 = time.time()
    if args.workers <= 1:
        for i, (key, url, dest) in enumerate(todo, 1):
            ok = _fetch(url, dest)
            n_ok += int(ok); n_fail += int(not ok)
            if i % 50 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                      f"rate={i / max(time.time() - t0, 1e-3):.2f}/s")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fetch, url, dest): (key, url, dest)
                       for key, url, dest in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                ok = fut.result()
                n_ok += int(ok); n_fail += int(not ok)
                if i % 50 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                          f"rate={i / max(time.time() - t0, 1e-3):.2f}/s")

    print(f"\ndone: {n_ok} downloaded, {n_fail} failed in {time.time() - t0:.0f}s")
    print(f"cache size: {sum(p.stat().st_size for p in cache_dir.iterdir() if p.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
