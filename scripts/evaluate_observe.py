"""
scripts/evaluate_observe.py
============================
Evaluate a trained OBSERVE checkpoint on a held-out slice of Phase 0R
trace JSONL.

Reports:
    routing_accuracy        argmax(routing_logits) == path[i+1]
    backtrack_accuracy      sigmoid(backtrack_logit) > 0.5  == target
    backtrack_f1            same, F1
    kappa_mae               | sigmoid(confidence_logit) - kappa_scalar |
    kappa_ece               expected calibration error (10 bins)
    oc_accuracy             overconfidence head accuracy

Outputs a JSON file with per-class routing breakdown + the aggregates.

Usage:
    python scripts/evaluate_observe.py \\
        --ckpt   observe/checkpoints/observe_best.pt \\
        --traces artifacts/observe_traces/phase0r_traces.jsonl \\
        --out    results/observe_eval.json \\
        --held-frac 0.1
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
    p.add_argument("--ckpt", required=True, help="OBSERVE checkpoint .pt")
    p.add_argument("--traces", required=True,
                   help="Phase 0R trace JSONL (same split seed as training!)")
    p.add_argument("--out", default="results/observe_eval.json")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--held-frac", type=float, default=0.1)
    p.add_argument("--val-frac",  type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--n-bins", type=int, default=10,
                   help="ECE bin count")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import numpy as np

    from observe.model import OBSERVE, AGENT_CLASSES_DEFAULT
    from observe.trainer import (
        OBSERVETrainer,
        load_phase0r_traces,
        split_annotations,
    )
    from observe.decision_transformer import _ece

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"=== evaluate_observe ===")
    print(f"  ckpt:    {args.ckpt}")
    print(f"  traces:  {args.traces}")
    print(f"  device:  {device}")

    if not Path(args.traces).is_file():
        raise SystemExit(f"traces not found: {args.traces}")

    # ---- load + split (same seed as training to get the held-out fold) ----
    annotations = load_phase0r_traces(args.traces)
    splits = split_annotations(
        annotations, val_frac=args.val_frac, held_frac=args.held_frac, seed=args.seed,
    )
    held = splits["held"]
    print(f"  held annotations: {len(held)} ({splits['n_images_held']} images)")
    if not held:
        raise SystemExit("held set is empty — try a larger --held-frac")

    # ---- model ----
    print(f"  building OBSERVE on {args.backbone} ...")
    model = OBSERVE(
        backbone=args.backbone,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # ---- loader ----
    trainer = OBSERVETrainer(model, lr=1e-4)
    loader = trainer.make_loader(held, args.batch_size, shuffle=False)

    # ---- collect predictions ----
    all_pred_agent = []
    all_true_agent = []
    all_pred_bt    = []
    all_true_bt    = []
    all_pred_k     = []
    all_true_k     = []
    all_pred_oc    = []
    all_true_oc    = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in batch["inputs"].items()}
            labels = batch["labels"]
            out = model(inputs)
            all_pred_agent.extend(out["routing_logits"].argmax(dim=-1).cpu().tolist())
            all_true_agent.extend(labels["next_agent"].tolist())
            all_pred_bt.extend(torch.sigmoid(out["backtrack_logit"]).cpu().tolist())
            all_true_bt.extend(labels["backtrack"].tolist())
            all_pred_k.extend(torch.sigmoid(out["confidence_logit"]).cpu().tolist())
            all_true_k.extend(labels["confidence"].tolist())
            all_pred_oc.extend(torch.sigmoid(out["oc_logit"]).cpu().tolist())
            all_true_oc.extend(labels["overconfidence"].tolist())

    # ---- metrics ----
    n = len(all_pred_agent)

    routing_acc = sum(1 for p, t in zip(all_pred_agent, all_true_agent) if p == t) / max(n, 1)

    per_class = {}
    for cls_idx, cls_name in enumerate(AGENT_CLASSES_DEFAULT):
        idxs = [i for i, t in enumerate(all_true_agent) if t == cls_idx]
        if not idxs:
            continue
        hits = sum(1 for i in idxs if all_pred_agent[i] == cls_idx)
        per_class[cls_name] = {
            "support": len(idxs),
            "accuracy": hits / len(idxs),
        }

    # backtrack
    bt_pred_bin = [1 if p > 0.5 else 0 for p in all_pred_bt]
    bt_true_bin = [int(t > 0.5) for t in all_true_bt]
    tp = sum(1 for p, t in zip(bt_pred_bin, bt_true_bin) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(bt_pred_bin, bt_true_bin) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(bt_pred_bin, bt_true_bin) if p == 0 and t == 1)
    bt_acc = sum(1 for p, t in zip(bt_pred_bin, bt_true_bin) if p == t) / max(n, 1)
    bt_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    # kappa calibration
    kappa_mae = sum(abs(p - t) for p, t in zip(all_pred_k, all_true_k)) / max(n, 1)
    # ECE: "correct" = within 0.15 of target (proxy since kappa is a regression target).
    correct = [abs(p - t) < 0.15 for p, t in zip(all_pred_k, all_true_k)]
    kappa_ece = _ece(all_pred_k, correct, n_bins=args.n_bins)

    # OC head
    oc_pred_bin = [1 if p > 0.5 else 0 for p in all_pred_oc]
    oc_true_bin = [int(t > 0.5) for t in all_true_oc]
    oc_acc = sum(1 for p, t in zip(oc_pred_bin, oc_true_bin) if p == t) / max(n, 1)

    results = {
        "n_samples":        n,
        "n_images":         splits["n_images_held"],
        "routing_accuracy": routing_acc,
        "routing_per_class": per_class,
        "backtrack_accuracy": bt_acc,
        "backtrack_f1":     bt_f1,
        "kappa_mae":        kappa_mae,
        "kappa_ece":        kappa_ece,
        "overconfidence_accuracy": oc_acc,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print(f"=== results (n={n}) ===")
    print(f"  routing_accuracy:       {routing_acc:.3f}")
    print(f"  backtrack_accuracy:     {bt_acc:.3f}  F1: {bt_f1:.3f}")
    print(f"  kappa_mae:              {kappa_mae:.3f}")
    print(f"  kappa_ece (10 bins):    {kappa_ece:.3f}")
    print(f"  overconfidence_acc:     {oc_acc:.3f}")
    print(f"  per-class routing:")
    for cls, d in per_class.items():
        print(f"    {cls:18s}  support={d['support']:4d}  acc={d['accuracy']:.3f}")
    print()
    print(f"  results → {out_path}")


if __name__ == "__main__":
    main()
