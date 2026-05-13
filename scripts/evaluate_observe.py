"""
scripts/evaluate_observe.py
============================
Evaluate a trained OBSERVE checkpoint on a held-out slice of Phase 0R
per-pass trace JSONL.

After Algorithm-1 routing removal, the student only predicts
uncertainty. The eval reports:
    kappa_mae       | sigmoid(confidence_logit) - kappa_scalar |
    kappa_ece       expected calibration error (10 bins)
    oc_accuracy     overconfidence head accuracy
    epistemic_mae   | sigmoid(epistemic_logit) - target |
    aleatoric_mae   | sigmoid(aleatoric_logit) - target |

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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", default="results/observe_eval.json")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--held-frac", type=float, default=0.1)
    p.add_argument("--val-frac",  type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()


def _ece(probs, correct, n_bins=10):
    if not probs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, c in zip(probs, correct):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx].append((p, c))
    ece = 0.0
    n = len(probs)
    for b in bins:
        if not b:
            continue
        avg_p   = sum(p for p, _ in b) / len(b)
        avg_acc = sum(1 for _, c in b if c) / len(b)
        ece += (len(b) / n) * abs(avg_p - avg_acc)
    return ece


def main() -> None:
    args = parse_args()
    import torch

    from observe.model import OBSERVE
    from observe.trainer import (
        OBSERVETrainer, load_phase0r_traces, split_annotations,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"=== evaluate_observe ===")
    print(f"  ckpt:    {args.ckpt}")
    print(f"  traces:  {args.traces}")
    print(f"  device:  {device}")

    if not Path(args.traces).is_file():
        raise SystemExit(f"traces not found: {args.traces}")

    annotations = load_phase0r_traces(args.traces)
    splits = split_annotations(
        annotations, val_frac=args.val_frac,
        held_frac=args.held_frac, seed=args.seed,
    )
    held = splits["held"]
    print(f"  held annotations: {len(held)} ({splits['n_images_held']} images)")
    if not held:
        raise SystemExit("held set is empty — try a larger --held-frac")

    print(f"  building OBSERVE on {args.backbone} ...")
    model = OBSERVE(
        backbone=args.backbone,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    trainer = OBSERVETrainer(model, lr=1e-4)
    loader = trainer.make_loader(held, args.batch_size, shuffle=False)

    all_pred_k     = []
    all_true_k     = []
    all_pred_eps   = []
    all_true_eps   = []
    all_pred_alpha = []
    all_true_alpha = []
    all_pred_oc    = []
    all_true_oc    = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in batch["inputs"].items()}
            labels = batch["labels"]
            out = model(inputs)
            all_pred_k.extend(torch.sigmoid(out["confidence_logit"]).cpu().tolist())
            all_true_k.extend(labels["confidence"].tolist())
            all_pred_eps.extend(torch.sigmoid(out["epistemic_logit"]).cpu().tolist())
            all_true_eps.extend(labels["epistemic"].tolist())
            all_pred_alpha.extend(torch.sigmoid(out["aleatoric_logit"]).cpu().tolist())
            all_true_alpha.extend(labels["aleatoric"].tolist())
            all_pred_oc.extend(torch.sigmoid(out["oc_logit"]).cpu().tolist())
            all_true_oc.extend(labels["overconfidence"].tolist())

    n = len(all_pred_k)

    def _mae(a, b):
        return sum(abs(x - y) for x, y in zip(a, b)) / max(n, 1)

    kappa_mae     = _mae(all_pred_k,     all_true_k)
    epistemic_mae = _mae(all_pred_eps,   all_true_eps)
    aleatoric_mae = _mae(all_pred_alpha, all_true_alpha)

    correct = [abs(p - t) < 0.15 for p, t in zip(all_pred_k, all_true_k)]
    kappa_ece = _ece(all_pred_k, correct, n_bins=args.n_bins)

    oc_pred_bin = [1 if p > 0.5 else 0 for p in all_pred_oc]
    oc_true_bin = [int(t > 0.5)        for t in all_true_oc]
    oc_acc = sum(1 for p, t in zip(oc_pred_bin, oc_true_bin) if p == t) / max(n, 1)

    results = {
        "n_samples":              n,
        "n_images":               splits["n_images_held"],
        "kappa_mae":              kappa_mae,
        "kappa_ece":              kappa_ece,
        "epistemic_mae":          epistemic_mae,
        "aleatoric_mae":          aleatoric_mae,
        "overconfidence_accuracy": oc_acc,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print(f"=== results (n={n}) ===")
    print(f"  kappa_mae:               {kappa_mae:.3f}")
    print(f"  kappa_ece (10 bins):     {kappa_ece:.3f}")
    print(f"  epistemic_mae:           {epistemic_mae:.3f}")
    print(f"  aleatoric_mae:           {aleatoric_mae:.3f}")
    print(f"  overconfidence_acc:      {oc_acc:.3f}")
    print(f"  results -> {out_path}")


if __name__ == "__main__":
    main()
