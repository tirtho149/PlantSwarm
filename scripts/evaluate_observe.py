"""
scripts/evaluate_observe.py
============================
Evaluate the trained OBSERVE classifier on PlantVillage and/or PlantWild
(Tomato by default).

Reports per dataset:
    n_samples
    top-1 accuracy (overall + per class)
    top-5 accuracy
    macro F1
    confusion matrix (saved as JSON)
    fraction of test classes with a KB prototype (the rest are
        scored zero-shot via the test-time synthetic class prompt)

PV folder layout expected: ``<root>/Tomato___Early_blight/*.jpg`` etc.
PW folder layout expected: ``<root>/tomato_early_blight/*.jpg`` etc.
(both forms are accepted by the loaders).

For classes in PV/PW that ARE NOT in the trained class index, we
build a one-line zero-shot prototype on the fly:

    "A field photograph of {crop} affected by {disease}."

This is the open-vocabulary case — the model never trained on the
disease but can still score it via the KB-augmented text geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", required=True,
                   help="OBSERVE checkpoint .pt produced by train_observe.py")
    p.add_argument("--pv-root", default=None,
                   help="PlantVillage root (folder per class). Optional.")
    p.add_argument("--pw-root", default=None,
                   help="PlantWild root (folder per class). Optional.")
    p.add_argument("--crop", default="Tomato",
                   help="Crop to evaluate (folders for other crops are skipped)")
    p.add_argument("--pv-classes-json", default="data/pv_classes.json",
                   help="Canonical PV class list for prototype synthesis")
    p.add_argument("--out", default="results/observe_eval.json")
    p.add_argument("--backbone", default="google/siglip-base-patch16-224")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="Cap test images per class (useful for sanity runs)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _build_extended_class_index(
    ckpt_labels,
    pv_root,
    crop,
    pv_classes_json,
):
    """Union the train-time class labels with PV's actual classes for
    this crop. New PV classes get synthesised one-line prototypes."""
    from observe.dataset import ClassIndex
    from observe.prototypes import build_healthy_prototype

    labels = list(ckpt_labels)
    proto_texts: list = []  # will be filled by the caller using the ckpt's stored prototypes
    new_synth: list = []

    if not pv_root:
        return labels, new_synth

    pv_meta = {}
    if pv_classes_json and Path(pv_classes_json).is_file():
        pv_meta = json.loads(Path(pv_classes_json).read_text())

    for c in pv_meta.get("classes", []):
        if c["crop"] != crop:
            continue
        label = f"{c['crop']}::{c['disease']}"
        if label in labels:
            continue
        # Synthesise a minimal prototype.
        if c.get("kind") == "healthy" or c["disease"] == "healthy":
            text = build_healthy_prototype(c["crop"])
        else:
            text = f"A field photograph of {c['crop']} affected by {c['disease']}."
        labels.append(label)
        new_synth.append({"label": label, "prototype": text})
    return labels, new_synth


def main() -> None:
    args = parse_args()
    import torch

    from observe.model import OBSERVE
    from observe.dataset import ClassIndex, PVFolderDataset, PWFolderDataset
    from observe.trainer import ImageCollator, encode_class_prototypes

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not Path(args.ckpt).is_file():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    print(f"=== evaluate_observe ===")
    print(f"  ckpt:    {args.ckpt}")
    print(f"  pv_root: {args.pv_root}")
    print(f"  pw_root: {args.pw_root}")
    print(f"  crop:    {args.crop}")
    print(f"  device:  {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_labels    = ckpt.get("class_labels") or []
    train_prototypes = ckpt.get("prototype_texts") or []
    if not train_labels:
        raise SystemExit("ckpt has no class_labels — was it produced by OBSERVETrainer.save()?")

    # Extend label set with any PV classes not seen at train time.
    extended_labels, new_synth = _build_extended_class_index(
        train_labels, args.pv_root, args.crop, args.pv_classes_json,
    )
    extended_prototypes = list(train_prototypes) + [s["prototype"] for s in new_synth]
    n_train = len(train_labels)
    n_synth = len(new_synth)
    print()
    print(f"  train-time KB prototypes : {n_train}")
    print(f"  zero-shot synth (new PV) : {n_synth}")
    print(f"  total classes at eval    : {len(extended_labels)}")
    for s in new_synth:
        print(f"    + synth: {s['label']}")

    # Build the model + load state.
    model = OBSERVE(
        backbone=args.backbone, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    proto_embeds = encode_class_prototypes(model, extended_prototypes, device=device)

    class_index = ClassIndex(extended_labels)

    results = {"crop": args.crop, "evals": {}}

    for tag, root, dataset_cls in (
        ("plantvillage", args.pv_root, PVFolderDataset),
        ("plantwild",    args.pw_root, PWFolderDataset),
    ):
        if not root:
            continue
        if not Path(root).is_dir():
            print(f"  [{tag}] root not found: {root} — skipping")
            continue
        print(f"\n--- {tag} ---")
        ds = dataset_cls(
            root=root, class_index=class_index, crop=args.crop,
            limit_per_class=args.limit_per_class,
        )
        stats = ds.stats()
        print(f"  samples: {stats['n_samples']}")
        if stats["n_samples"] == 0:
            results["evals"][tag] = {**stats, "note": "no test samples"}
            continue
        for k, v in stats.get("per_class", {}).items():
            print(f"    {k:38s} {v}")

        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=ImageCollator(model.processor),
        )

        per_class_correct = Counter()
        per_class_count   = Counter()
        per_class_pred    = Counter()       # for macro F1 calc
        per_class_topk    = defaultdict(int)
        confusion         = defaultdict(Counter)
        n = 0
        top1_correct = 0
        top5_correct = 0

        with torch.no_grad():
            for batch in loader:
                pixels = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                logits = model(pixels, proto_embeds)
                preds  = logits.argmax(dim=-1)
                top5   = logits.topk(min(5, logits.size(-1)), dim=-1).indices
                n += labels.size(0)
                top1_correct += int((preds == labels).sum().item())
                top5_correct += int((top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item())
                for true_id, pred_id in zip(labels.tolist(), preds.tolist()):
                    true_lbl = extended_labels[true_id]
                    pred_lbl = extended_labels[pred_id]
                    per_class_count[true_lbl] += 1
                    per_class_pred[pred_lbl]  += 1
                    if true_id == pred_id:
                        per_class_correct[true_lbl] += 1
                    confusion[true_lbl][pred_lbl] += 1

        # Macro F1 across the classes that appeared in this test set.
        f1s = []
        for lbl in per_class_count:
            tp = per_class_correct[lbl]
            fn = per_class_count[lbl] - tp
            fp = per_class_pred[lbl]  - tp
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        macro_f1 = sum(f1s) / max(1, len(f1s))

        eval_out = {
            "n_samples":     n,
            "top1_accuracy": top1_correct / max(1, n),
            "top5_accuracy": top5_correct / max(1, n),
            "macro_f1":      macro_f1,
            "per_class": {
                lbl: {
                    "support":  per_class_count[lbl],
                    "correct":  per_class_correct[lbl],
                    "accuracy": per_class_correct[lbl] / max(1, per_class_count[lbl]),
                    "in_kb":    lbl in set(train_labels),  # vs zero-shot synth
                }
                for lbl in per_class_count
            },
            "confusion": {k: dict(v) for k, v in confusion.items()},
        }
        results["evals"][tag] = eval_out

        print(f"  top1: {eval_out['top1_accuracy']:.3f}")
        print(f"  top5: {eval_out['top5_accuracy']:.3f}")
        print(f"  macro F1: {macro_f1:.3f}")
        print(f"  per-class accuracy:")
        for lbl, pc in sorted(eval_out["per_class"].items()):
            kb = "KB" if pc["in_kb"] else "zero-shot"
            print(f"    [{kb:9s}] {lbl:38s} acc={pc['accuracy']:.3f}  n={pc['support']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  results -> {out_path}")


if __name__ == "__main__":
    main()
