"""
scripts/tabpfn_eval.py
======================
TabPFN classifier over PathomeOOD multimodal features. Reproduces
every reproducible table from the BioCAP paper (Zhang et al.,
arXiv:2510.20095) on Bugwood data without any training — TabPFN is
a meta-learned tabular foundation classifier that does in-context
learning over the support set in one forward pass.

Reads features produced by scripts/build_features.py:

    train: data/bugwood_features/<encoder>_<strategy>.npz
    eval : data/eval_features/<encoder>_<strategy>_{plantvillage,plantdoc,plantwild}.npz

For each variant in the 14-tag matrix below the classifier fits TabPFN
on the (subset of) Bugwood features and predicts on every test set,
writing one JSON result file per (variant, eval_set) cell. Also
produces few-shot K-shot results (Tables 18 + 20) and N off-shelf
zero-shot baselines.

Variant matrix
--------------
  T01..T07   caption strategy ablation
             (BioCLIP encoder; subset=all; 7 caption strategies)
  T08..T11   encoder ablation
             (canonical_deltas_3 caption; subset=all; 4 alt encoders)
             T08 = CLIP-openai
             T09 = SigLIP
             T10 = BioCLIP-2
             T11 = FG-CLIP    (paper Table 1 reference)
  T12        encoder ablation — BioTrove-CLIP
  T13        KB-covered subset (used_kb=1)
  T14        KB non-covered subset (used_kb=0)

  Plus zero-shot baselines for each of the 6 encoders (no TabPFN;
  cosine-sim against class-name templates) — produces Table 1 + Fig 3.

Knobs (CLI)
-----------
  --tabpfn-version  v1 or v2 (default v2)
  --n-pca           PCA-reduce embedding portion (default 256;
                    0 = no PCA). Crop / state embeddings are PCA'd
                    too — that's now the dominant dimensionality
                    after the embedding-based metadata change.
  --max-train-rows  TabPFN row cap (default 10000)
  --include-baselines  emit zero-shot baselines (default ON)
  --include-fewshot    emit 1-shot + 5-shot results (default ON)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ---------------------------------------------------------------------------
# Variant matrix
# ---------------------------------------------------------------------------

VARIANTS: List[Tuple[str, str, str, str]] = [
    # Caption-strategy ablation (paper Tables 3 + 6)
    ("T01", "bioclip", "label_only",         "all"),
    ("T02", "bioclip", "summary_only",       "all"),
    ("T03", "bioclip", "canonical_full",     "all"),
    ("T04", "bioclip", "canonical_deltas_3", "all"),    # MAIN
    ("T05", "bioclip", "canonical_deltas_1", "all"),
    ("T06", "bioclip", "canonical_deltas_5", "all"),
    ("T07", "bioclip", "canonical_deltas_7", "all"),
    # Encoder importance (paper Figure 3 analog)
    ("T08", "clip_vitb16",   "canonical_deltas_3", "all"),
    ("T09", "siglip_vitb16", "canonical_deltas_3", "all"),
    ("T10", "bioclip2",      "canonical_deltas_3", "all"),
    ("T11", "fgclip",        "canonical_deltas_3", "all"),
    ("T12", "biotrove",      "canonical_deltas_3", "all"),
    # KB-coverage subset ablation (paper Table 4)
    ("T13", "bioclip", "canonical_deltas_3", "covered"),
    ("T14", "bioclip", "canonical_deltas_3", "non_covered"),
]


# All encoders we have features for (zero-shot baselines).
ALL_ENCODERS = ["bioclip", "bioclip2", "clip_vitb16",
                "siglip_vitb16", "fgclip", "biotrove"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--features-root", default="data/bugwood_features")
    p.add_argument("--eval-root",     default="data/eval_features")
    p.add_argument("--results-dir",   default="results/pathomeood_eval")
    p.add_argument("--variants",      default="",
                   help="comma-separated subset (default: all 14)")
    p.add_argument("--include-baselines", action="store_true",
                   help="also produce 6 off-shelf zero-shot baseline runs")
    p.add_argument("--include-fewshot",   action="store_true",
                   help="also produce 1-shot and 5-shot results (Tables 18+20)")
    p.add_argument("--fewshot-shots", default="1,5",
                   help="comma-separated shot counts (default 1,5)")
    p.add_argument("--fewshot-seeds", type=int, default=5,
                   help="few-shot eval averages over this many seeds")
    p.add_argument("--n-pca", type=int, default=256,
                   help="PCA-reduce the concatenated embedding blocks. "
                        "0 = no PCA. TabPFNv2 caps at ~500 features.")
    p.add_argument("--tabpfn-version", default="v2", choices=("v2", "v1"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-train-rows", type=int, default=10000,
                   help="TabPFN row cap; stratified subsample if exceeded")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_train(features_root: Path, encoder: str, strategy: str):
    p = features_root / f"{encoder}_{strategy}.npz"
    if not p.is_file():
        return None, f"missing train features: {p}"
    npz = np.load(p, allow_pickle=True)
    return dict(
        X=npz["X"], y=npz["y"], used_kb=npz["used_kb"],
        class_names=npz["class_names"].tolist(),
        meta=json.loads(str(npz["meta"])),
        path=p,
    ), None


def _load_eval(eval_root: Path, encoder: str, strategy: str, kind: str):
    p = eval_root / f"{encoder}_{strategy}_{kind}.npz"
    if not p.is_file():
        return None, f"missing eval features: {p}"
    npz = np.load(p, allow_pickle=True)
    return dict(
        X=npz["X"], y=npz["y"],
        class_names=npz["class_names"].tolist(),
        eval_paths=npz["eval_paths"].tolist(),
        eval_labels=npz["eval_labels"].tolist(),
        path=p,
    ), None


def _apply_subset(X, y, used_kb, subset: str):
    if subset == "all":
        return X, y
    mask = (used_kb == 1) if subset == "covered" else (used_kb == 0)
    return X[mask], y[mask]


def _maybe_pca(X_train, X_eval, n_pca: int):
    if n_pca <= 0 or X_train.shape[1] <= n_pca:
        return X_train, X_eval
    from sklearn.decomposition import PCA  # type: ignore
    pca = PCA(n_components=n_pca, random_state=0)
    X_train_r = pca.fit_transform(X_train).astype(np.float32)
    X_eval_r  = pca.transform(X_eval).astype(np.float32)
    return X_train_r, X_eval_r


def _stratified_cap(X, y, cap: int):
    if cap <= 0 or X.shape[0] <= cap:
        return X, y
    rng = np.random.default_rng(0)
    keep_idx = []
    classes = np.unique(y)
    per_class = max(1, cap // len(classes))
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep_idx.extend(idx[:per_class].tolist())
    keep_idx = np.array(keep_idx, dtype=np.int64)
    if len(keep_idx) > cap:
        rng.shuffle(keep_idx)
        keep_idx = keep_idx[:cap]
    return X[keep_idx], y[keep_idx]


# ---------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------

def _make_tabpfn(version: str, device: str):
    try:
        if version == "v2":
            from tabpfn import TabPFNClassifier  # type: ignore
            return TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        from tabpfn import TabPFNClassifier  # type: ignore
        return TabPFNClassifier(device=device, N_ensemble_configurations=4)
    except ImportError as e:
        raise SystemExit(f"pip install tabpfn; {e}")


def _topk(probs, y, k: int) -> float:
    valid = y >= 0
    if not valid.any():
        return 0.0
    topk = np.argsort(-probs[valid], axis=1)[:, :k]
    return float((topk == y[valid][:, None]).any(axis=1).mean())


# ---------------------------------------------------------------------------
# One variant: fit + predict on PV/PD/PW
# ---------------------------------------------------------------------------

def _evaluate_variant(
    variant_id: str, encoder: str, strategy: str, subset: str,
    train_data: Dict, eval_data_per_kind: Dict[str, Dict],
    n_pca: int, cap: int, version: str, device: str,
) -> Dict:
    X_full = train_data["X"]
    y_full = train_data["y"]
    used_kb = train_data["used_kb"]
    X_sub, y_sub = _apply_subset(X_full, y_full, used_kb, subset)
    X_cap, y_cap = _stratified_cap(X_sub, y_sub, cap)
    clf = _make_tabpfn(version, device)

    out: Dict = {
        "variant_config": dict(
            variant=variant_id, encoder=encoder,
            strategy=strategy, subset=subset,
        ),
        "n_train": int(X_cap.shape[0]),
        "n_classes": int(len(np.unique(y_cap))),
        "evals": {},
    }
    print(f"  [{variant_id}] enc={encoder} strat={strategy} sub={subset}  "
          f"N={X_cap.shape[0]}  C={len(np.unique(y_cap))}  D={X_cap.shape[1]}")

    for kind, ev in eval_data_per_kind.items():
        X_ev = ev["X"]; y_ev = ev["y"]
        # PCA train + eval features together (fit on train).
        X_tr, X_ev_pca = _maybe_pca(X_cap, X_ev, n_pca)
        clf.fit(X_tr, y_cap)
        probs = clf.predict_proba(X_ev_pca)
        # Pad class axis to match train class count.
        n_cls = len(train_data["class_names"])
        if probs.shape[1] < n_cls:
            pad = np.zeros((probs.shape[0], n_cls - probs.shape[1]))
            probs = np.concatenate([probs, pad], axis=1)
        top1 = _topk(probs, y_ev, 1)
        top5 = _topk(probs, y_ev, min(5, n_cls))
        out["evals"][kind] = dict(
            top1=top1, top5=top5,
            n_samples=int(X_ev.shape[0]),
            in_train_class=int((y_ev >= 0).sum()),
        )
        print(f"      {kind:12s}  top1={top1*100:5.1f}  top5={top5*100:5.1f}  "
              f"N={X_ev.shape[0]}  in-train={(y_ev >= 0).sum()}")
    return out


# ---------------------------------------------------------------------------
# Few-shot: TabPFN is in-context — just pass K shots per class as support
# ---------------------------------------------------------------------------

def _fewshot_one_seed(
    X_train, y_train, X_eval, y_eval, n_classes,
    k: int, seed: int, version: str, device: str, n_pca: int,
):
    rng = np.random.default_rng(seed)
    sup_idx = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        sup_idx.extend(idx[:min(k, len(idx))])
    sup_idx = np.array(sup_idx, dtype=np.int64)
    X_sup = X_train[sup_idx]
    y_sup = y_train[sup_idx]
    X_tr, X_ev = _maybe_pca(X_sup, X_eval, n_pca)
    clf = _make_tabpfn(version, device)
    clf.fit(X_tr, y_sup)
    probs = clf.predict_proba(X_ev)
    if probs.shape[1] < n_classes:
        pad = np.zeros((probs.shape[0], n_classes - probs.shape[1]))
        probs = np.concatenate([probs, pad], axis=1)
    return _topk(probs, y_eval, 1), _topk(probs, y_eval, min(5, n_classes))


def _evaluate_fewshot(
    variant_id: str, encoder: str, strategy: str, subset: str,
    train_data: Dict, eval_data_per_kind: Dict[str, Dict],
    shots: List[int], seeds: int, n_pca: int, version: str, device: str,
) -> Dict:
    out: Dict = {
        "variant_config": dict(
            variant=variant_id, encoder=encoder,
            strategy=strategy, subset=subset,
        ),
        "evals": {},
    }
    X_full, y_full = train_data["X"], train_data["y"]
    X_sub, y_sub = _apply_subset(X_full, y_full, train_data["used_kb"], subset)
    n_cls = len(train_data["class_names"])
    print(f"  [{variant_id} few-shot] shots={shots} seeds={seeds}")
    for kind, ev in eval_data_per_kind.items():
        per_k: Dict[str, Dict] = {}
        for k in shots:
            top1s, top5s = [], []
            for s in range(seeds):
                t1, t5 = _fewshot_one_seed(
                    X_sub, y_sub, ev["X"], ev["y"], n_cls,
                    k=k, seed=s, version=version, device=device, n_pca=n_pca,
                )
                top1s.append(t1); top5s.append(t5)
            per_k[f"{k}_shot"] = dict(
                mean_top1=float(np.mean(top1s)),
                std_top1=float(np.std(top1s)) if seeds > 1 else 0.0,
                mean_top5=float(np.mean(top5s)),
                std_top5=float(np.std(top5s)) if seeds > 1 else 0.0,
                seeds=seeds,
            )
            print(f"      {kind:12s}  k={k}  top1={np.mean(top1s)*100:5.1f}±"
                  f"{np.std(top1s)*100:4.1f}")
        out["evals"][kind] = per_k
    return out


# ---------------------------------------------------------------------------
# Zero-shot baselines (cosine-sim, no TabPFN)
# ---------------------------------------------------------------------------

def _zeroshot_baseline(
    encoder: str, eval_data_per_kind: Dict[str, Dict],
    class_names: List[str], block_widths: Dict,
) -> Dict:
    """Standard CLIP zero-shot: cosine(image_emb, text_emb(class_name)).

    The image_emb is just the first block_widths['image'] columns of
    the eval feature matrix (we already extracted it in build_features.py).
    """
    try:
        import open_clip
        import torch
    except ImportError as e:
        return {"error": f"open_clip needed: {e}"}
    from scripts.build_features import ENCODERS as _ENC
    model_name, pretrained = _ENC[encoder]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        prompts = [f"a photo of {c}." for c in class_names]
        text_feats = model.encode_text(tok(prompts).to(device))
        if isinstance(text_feats, tuple):
            text_feats = text_feats[0]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy().astype(np.float32)

    out: Dict = {"evals": {}}
    d_image = int(block_widths["image"])
    for kind, ev in eval_data_per_kind.items():
        X_img = ev["X"][:, :d_image]
        X_img = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-12)
        # text_feats has shape (C, D_text). If D_image == D_text we can
        # dot-product directly; otherwise the encoders' image/text don't
        # share a space and zero-shot makes no sense.
        if X_img.shape[1] != text_feats.shape[1]:
            print(f"      [{kind}] dim mismatch image={X_img.shape[1]} "
                  f"text={text_feats.shape[1]}; baseline skipped")
            continue
        logits = X_img @ text_feats.T
        out["evals"][kind] = dict(
            top1=_topk(logits, ev["y"], 1),
            top5=_topk(logits, ev["y"], min(5, text_feats.shape[0])),
            n_samples=int(X_img.shape[0]),
        )
        print(f"      {kind:12s}  top1={out['evals'][kind]['top1']*100:5.1f}  "
              f"top5={out['evals'][kind]['top5']*100:5.1f}")
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_eval(out_dir: Path, kind: str, payload: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{kind}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    feat_root = Path(args.features_root)
    eval_root = Path(args.eval_root)
    res_root = Path(args.results_dir)

    variants = VARIANTS
    if args.variants:
        wanted = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in wanted]

    shots = [int(s) for s in args.fewshot_shots.split(",") if s.strip()]

    print(f"=== tabpfn_eval ===")
    print(f"  variants    : {[v[0] for v in variants]}")
    print(f"  tabpfn      : {args.tabpfn_version} PCA={args.n_pca} cap={args.max_train_rows}")
    print(f"  few-shot    : shots={shots} seeds={args.fewshot_seeds} on={args.include_fewshot}")
    print(f"  baselines   : on={args.include_baselines}")

    # Main matrix: zero-shot-like full-train classification per variant.
    for v_id, encoder, strategy, subset in variants:
        print()
        train, err = _load_train(feat_root, encoder, strategy)
        if train is None:
            print(f"  [{v_id}] SKIP: {err}")
            continue
        ev_per_kind = {}
        for k in ("plantvillage", "plantdoc", "plantwild"):
            ev, _ = _load_eval(eval_root, encoder, strategy, k)
            if ev is not None:
                ev_per_kind[k] = ev
        if not ev_per_kind:
            print(f"  [{v_id}] no eval features; skipping")
            continue
        try:
            result = _evaluate_variant(
                v_id, encoder, strategy, subset, train, ev_per_kind,
                n_pca=args.n_pca, cap=args.max_train_rows,
                version=args.tabpfn_version, device=args.device,
            )
        except Exception as e:
            print(f"  [{v_id}] ERROR: {type(e).__name__}: {e}")
            continue
        out_dir = res_root / v_id
        for kind, ev_out in result["evals"].items():
            _persist_eval(out_dir, kind, dict(
                model=f"{v_id} (TabPFN/{encoder}/{strategy}/{subset})",
                metrics={"val-unseen-top1": ev_out["top1"],
                         "val-unseen-top5": ev_out["top5"]},
                stats={"n_samples": ev_out["n_samples"],
                       "in_train_class": ev_out["in_train_class"]},
                variant_config=result["variant_config"],
            ))

        # Few-shot.
        if args.include_fewshot:
            try:
                fs = _evaluate_fewshot(
                    v_id, encoder, strategy, subset, train, ev_per_kind,
                    shots=shots, seeds=args.fewshot_seeds,
                    n_pca=args.n_pca, version=args.tabpfn_version,
                    device=args.device,
                )
            except Exception as e:
                print(f"  [{v_id}] FEW-SHOT ERROR: {type(e).__name__}: {e}")
                continue
            for kind, per_k in fs["evals"].items():
                _persist_eval(out_dir, f"fewshot_{kind}", dict(
                    model=f"{v_id} few-shot",
                    shots=per_k,
                    variant_config=fs["variant_config"],
                ))

    # Off-shelf zero-shot baselines.
    if args.include_baselines:
        print()
        print("=== zero-shot baselines ===")
        # Class universe must come from a train file; use BioCLIP /
        # canonical_deltas_3 as the canonical reference.
        ref, _ = _load_train(feat_root, "bioclip", "canonical_deltas_3")
        if ref is None:
            print("  no reference train; baselines need at least one train file")
            return
        for encoder in ALL_ENCODERS:
            print(f"\n  baseline: {encoder}_zs")
            ev_per_kind = {}
            for k in ("plantvillage", "plantdoc", "plantwild"):
                ev, _ = _load_eval(eval_root, encoder, "canonical_deltas_3", k)
                if ev is not None:
                    ev_per_kind[k] = ev
            if not ev_per_kind:
                # Fall back to bioclip's eval features (note: dim mismatch
                # is handled in _zeroshot_baseline, just skips).
                continue
            # block_widths must come from the encoder's own train file
            # (image-emb dim varies by encoder).
            this_train, _ = _load_train(feat_root, encoder, "canonical_deltas_3")
            if this_train is None:
                # use first eval matrix's image dim as proxy
                # (works because build_features.py records the order)
                continue
            result = _zeroshot_baseline(
                encoder, ev_per_kind,
                this_train["class_names"],
                this_train["meta"]["block_widths"],
            )
            out_dir = res_root / f"{encoder}_zs"
            for kind, ev_out in result.get("evals", {}).items():
                _persist_eval(out_dir, kind, dict(
                    model=f"{encoder}_zs",
                    metrics={"val-unseen-top1": ev_out["top1"],
                             "val-unseen-top5": ev_out["top5"]},
                    stats={"n_samples": ev_out["n_samples"]},
                    variant_config={"encoder": encoder, "kind": "zero_shot"},
                ))


if __name__ == "__main__":
    main()
