"""
scripts/train_observe.py
========================
Train OBSERVE on Phase 0R trace JSONL (delta-mode supervision).

Expects traces produced by Phase 0R with PATHOME_TRACE_DIR set, e.g.
artifacts/observe_traces/phase0r_traces.jsonl.

Usage:
    python scripts/train_observe.py \
        --traces artifacts/observe_traces/phase0r_traces.jsonl \
        --save-dir observe/checkpoints/ \
        --epochs 5 --batch-size 4

For a full A100 run, see scripts/submit_observe_train.sh.
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
    p.add_argument("--traces", required=True,
                   help="Phase 0R trace JSONL path")
    p.add_argument("--save-dir", default="observe/checkpoints/")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--held-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cuda | cpu (default: cuda if available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy torch imports so --help works without GPU deps installed.
    import torch

    from observe.model import OBSERVE
    from observe.loss import ObserveLoss, ObserveLossWeights
    from observe.trainer import (
        OBSERVETrainer,
        load_phase0r_traces,
        split_annotations,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"=== train_observe ===")
    print(f"  traces:    {args.traces}")
    print(f"  save_dir:  {args.save_dir}")
    print(f"  device:    {device}")
    print(f"  backbone:  {args.backbone}")
    print(f"  LoRA r/α:  {args.lora_r} / {args.lora_alpha}")

    # ---- 1. Load + split ---------------------------------------------------
    if not Path(args.traces).is_file():
        raise SystemExit(f"traces not found: {args.traces}")
    annotations = load_phase0r_traces(args.traces)
    print(f"  loaded:    {len(annotations)} per-step annotations")
    if not annotations:
        raise SystemExit("No annotations extracted — is the trace JSONL non-empty?")

    splits = split_annotations(
        annotations,
        val_frac=args.val_frac,
        held_frac=args.held_frac,
        seed=args.seed,
    )
    print(f"  split:     train={len(splits['train'])}  "
          f"val={len(splits['val'])}  held={len(splits['held'])}")
    print(f"             ({splits['n_images_train']} / {splits['n_images_val']} / "
          f"{splits['n_images_held']} unique images)")

    # ---- 2. Build model + processor ----------------------------------------
    print(f"  building OBSERVE on {args.backbone} ...")
    model = OBSERVE(
        backbone=args.backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model = model.to(device)

    # ---- 3. Loss + trainer + loaders ---------------------------------------
    loss_fn = ObserveLoss(ObserveLossWeights()).to(device)
    trainer = OBSERVETrainer(model, lr=args.lr)
    train_loader = trainer.make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader   = trainer.make_loader(splits["val"],   args.batch_size, shuffle=False)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- epoch {epoch}/{args.epochs} ---")
        train_metrics = trainer.train_epoch(train_loader, loss_fn, device)
        val_metrics   = trainer.validate(val_loader, loss_fn, device)
        line = {
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
        }
        history.append(line)
        print(f"  train: {train_metrics}")
        print(f"  val:   {val_metrics}")

        if val_metrics.get("total", float("inf")) < best_val:
            best_val = val_metrics["total"]
            trainer.save(save_dir / "observe_best.pt")
            print(f"  ✓ new best (val.total = {best_val:.4f}) — saved")

    # Always save the last epoch too.
    trainer.save(save_dir / "observe_last.pt")
    with open(save_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nDone. Checkpoints: {save_dir}/observe_best.pt + observe_last.pt")
    print(f"     history:     {save_dir}/history.json")


if __name__ == "__main__":
    main()
