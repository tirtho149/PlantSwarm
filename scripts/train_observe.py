"""
scripts/train_observe.py
========================
Train the OBSERVE KB-augmented OOD classifier on Bugwood.

Default scope: Tomato only. The KB prototypes come from
``artifacts/pathome_seed/symptoms_seed.json`` (Phase 0 + Phase 0R
output); only Tomato profiles are picked up, plus a synthetic
"Tomato::healthy" prototype.

Training images come from ``BugWood_Diseases_usable.csv`` (filtered
to NormCrop == Tomato), with images read from
``.bugwood_cache/<image_number>.{jpg|png|webp}``. The image cache must
be populated first via ``scripts/setup_image_cache.sh``.

Usage:
    python scripts/train_observe.py \\
        --seed artifacts/pathome_seed/symptoms_seed.json \\
        --bugwood-csv BugWood_Diseases_usable.csv \\
        --cache-dir .bugwood_cache \\
        --crop Tomato \\
        --backbone google/siglip-base-patch16-224 \\
        --epochs 10 --batch-size 32 --lora-r 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", required=True,
                   help="symptoms_seed.json path")
    p.add_argument("--bugwood-csv", default="BugWood_Diseases_usable.csv")
    p.add_argument("--cache-dir", default=".bugwood_cache",
                   help="comma-separated cache search dirs")
    p.add_argument("--crop", default="Tomato")
    p.add_argument("--include-healthy", action="store_true",
                   help="add a synthetic '<crop>::healthy' class so the "
                        "model has a non-disease prototype at training time")
    p.add_argument("--save-dir", default="observe/checkpoints/")
    p.add_argument("--backbone", default="google/siglip-base-patch16-224")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed-rng", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from observe.model import OBSERVE
    from observe.prototypes import (
        add_healthy_prototypes, load_seed_prototypes,
    )
    from observe.dataset import BugwoodTomatoDataset, ClassIndex
    from observe.trainer import OBSERVETrainer, split_indices

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---- 1. Build the class index + prototypes from seed JSON ----------
    seed_path = Path(args.seed)
    if not seed_path.is_file():
        raise SystemExit(f"seed JSON not found: {seed_path}")
    proto_records = load_seed_prototypes(
        seed_path, crop_filter=args.crop, top_k_regional=3,
    )
    if args.include_healthy:
        proto_records = add_healthy_prototypes(proto_records, [args.crop])
    if not proto_records:
        raise SystemExit(
            f"No KB profiles found for crop={args.crop} in {seed_path}. "
            "Did Phase 0 / 0R run for this crop?"
        )

    class_labels = [r["label"]     for r in proto_records]
    proto_texts  = [r["prototype"] for r in proto_records]
    class_index  = ClassIndex(class_labels)

    print(f"=== train_observe (crop={args.crop}) ===")
    print(f"  KB prototypes loaded: {len(class_labels)}")
    print(f"  device:               {device}")
    print(f"  backbone:             {args.backbone}")
    print(f"  LoRA r/alpha:         {args.lora_r} / {args.lora_alpha}")
    print()
    for r in proto_records:
        preview = r["prototype"][:120].replace("\n", " ")
        print(f"   [{r['kind']:7s}] {r['label']:34s}  {preview}...")

    # ---- 2. Build dataset --------------------------------------------------
    cache_dirs = [d.strip() for d in args.cache_dir.split(",") if d.strip()]
    dataset = BugwoodTomatoDataset(
        csv_path=args.bugwood_csv,
        cache_dirs=cache_dirs,
        class_index=class_index,
        crop=args.crop,
    )
    print()
    print(f"=== dataset ===")
    for k, v in dataset.stats().items():
        print(f"  {k:24s} {v}")
    if len(dataset) == 0:
        raise SystemExit(
            "No images found. Make sure scripts/setup_image_cache.sh has "
            "populated the cache against the same filtered CSV."
        )

    splits = split_indices(dataset, val_frac=args.val_frac, seed=args.seed_rng)
    print(f"  train / val split: {len(splits['train'])} / {len(splits['val'])}")

    # ---- 3. Model + trainer ------------------------------------------------
    print()
    print(f"  building OBSERVE on {args.backbone} ...")
    model = OBSERVE(
        backbone=args.backbone,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ).to(device)

    trainer = OBSERVETrainer(
        model=model,
        class_index=class_index,
        prototype_texts=proto_texts,
        lr=args.lr,
    )
    train_loader = trainer.make_loader(dataset, splits["train"], args.batch_size, shuffle=True)
    val_loader   = trainer.make_loader(dataset, splits["val"],   args.batch_size, shuffle=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- epoch {epoch}/{args.epochs} ---")
        train_m = trainer.train_epoch(train_loader, device)
        val_m   = trainer.validate(val_loader,   device)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(f"  train: loss={train_m['loss']:.4f}  top1={train_m['top1']:.3f}")
        print(f"  val:   loss={val_m['loss']:.4f}  top1={val_m['top1']:.3f}")
        if val_m["top1"] > best_val:
            best_val = val_m["top1"]
            trainer.save(save_dir / "observe_best.pt")
            print(f"  + new best val top-1 = {best_val:.3f} — saved")

    trainer.save(save_dir / "observe_last.pt")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Checkpoints: {save_dir}/observe_best.pt + observe_last.pt")
    print(f"     history:     {save_dir}/history.json")


if __name__ == "__main__":
    main()
