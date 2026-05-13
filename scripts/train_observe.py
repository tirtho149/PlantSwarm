"""
scripts/train_observe.py
========================
Train OBSERVE on Phase 0R per-pass trace JSONL.

After Algorithm-1 routing removal, supervision is uncertainty-only
(epistemic / aleatoric / confidence / overconfidence). The student
learns to predict its own calibrated uncertainty over a single-pass
delta extraction.

Usage:
    python scripts/train_observe.py \\
        --traces artifacts/observe_traces/phase0r_traces.jsonl \\
        --save-dir observe/checkpoints/ \\
        --epochs 5 --batch-size 4
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
                   help="Phase 0R per-pass trace JSONL path")
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
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from observe.model import OBSERVE
    from observe.loss import ObserveLoss, ObserveLossWeights
    from observe.trainer import (
        OBSERVETrainer, load_phase0r_traces, split_annotations,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=== train_observe ===")
    print(f"  traces:    {args.traces}")
    print(f"  save_dir:  {args.save_dir}")
    print(f"  device:    {device}")
    print(f"  backbone:  {args.backbone}")
    print(f"  LoRA r/a:  {args.lora_r} / {args.lora_alpha}")

    if not Path(args.traces).is_file():
        raise SystemExit(f"traces not found: {args.traces}")
    annotations = load_phase0r_traces(args.traces)
    print(f"  loaded:    {len(annotations)} per-pass annotations")
    if not annotations:
        raise SystemExit("No annotations extracted.")

    splits = split_annotations(
        annotations,
        val_frac=args.val_frac, held_frac=args.held_frac, seed=args.seed,
    )
    print(f"  split:     train={len(splits['train'])}  "
          f"val={len(splits['val'])}  held={len(splits['held'])}")
    print(f"             ({splits['n_images_train']} / "
          f"{splits['n_images_val']} / {splits['n_images_held']} unique images)")

    print(f"  building OBSERVE on {args.backbone} ...")
    model = OBSERVE(
        backbone=args.backbone,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ).to(device)

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
        train_m = trainer.train_epoch(train_loader, loss_fn, device)
        val_m   = trainer.validate(val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(f"  train: {train_m}")
        print(f"  val:   {val_m}")
        if val_m.get("total", float("inf")) < best_val:
            best_val = val_m["total"]
            trainer.save(save_dir / "observe_best.pt")
            print(f"  + new best (val.total = {best_val:.4f}) — saved")
    trainer.save(save_dir / "observe_last.pt")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Checkpoints: {save_dir}/observe_best.pt + observe_last.pt")


if __name__ == "__main__":
    main()
