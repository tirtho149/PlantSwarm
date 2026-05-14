"""
scripts/build_features.py
=========================
Extract multimodal features for the TabPFN classifier step.

For each (encoder, caption_strategy) pair the script produces:

  data/bugwood_features/<encoder>_<strategy>.npz
      X_train         (N_train, D_total)    feature matrix
      y_train         (N_train,)            integer class labels
      class_names     (C,)                  list of "Crop Disease" strings
      used_kb         (N_train,)            1 if row had a real KB profile
                                            (vs fallback caption template);
                                            drives T10/T11 covered/non-covered
                                            subsets at TabPFN time
      block_widths    JSON                  per-block widths so the
                                            classifier can PCA the
                                            embedding portion separately
      crops_vocab     list                  ordered list of crop names
                                            (preserved for compatibility)
  data/eval_features/<encoder>_<strategy>_<dataset>.npz
      same shape; y_eval entries are -1 when the test class isn't in train

Feature vector layout (D_total ≈ 1500–2500):

    [ image_emb        | caption_emb       | crop_emb         | state_emb       ]
    (D_image ~512–1024)  (D_text ~512)       (D_text ~512)      (D_text ~512)

  - image_emb        frozen visual encoder on the image
  - caption_emb      paired text tower on the KB-derived caption
  - crop_emb         paired text tower on "a photograph of <crop> plant"
                     (EMBEDDING based, not one-hot — captures semantic
                      similarity between e.g. Tomato and Potato)
  - state_emb        paired text tower on "a photograph taken in <state>"
                     (Bugwood-only; at test time we use "unknown
                      location" which becomes a fixed sentinel embedding)

Why text embeddings instead of one-hot
--------------------------------------
- One-hot makes "Tomato" and "Cucumber" maximally distant in feature
  space (orthogonal), losing crop-family priors. Text embeddings put
  semantically related crops close together.
- One-hot can't generalize to test-set crops that weren't in training.
  Text embeddings can (the encoder knows what "Apple" means even if
  no Apple training images existed).
- Dimensional consistency: image_emb, caption_emb, crop_emb, state_emb
  all come from the same encoder family, so they live in compatible
  spaces (image-text alignment matters for the joint geometry).

Supported encoders (see ENCODERS below): bioclip, bioclip2, clip_vitb16,
siglip_vitb16, fgclip, biotrove. Add more by extending ENCODERS.

Usage
-----
  python scripts/build_features.py \\
      --captions data/bugwood_captions/all_canonical_deltas_3.parquet \\
      --encoder  bioclip \\
      --eval-pv  data/eval/PlantVillage \\
      --eval-pd  data/eval/PlantDoc/test \\
      --eval-pw  data/eval/PlantWild
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from plantswarm.captioning import (
    build_disease_caption, build_fallback_caption, build_healthy_caption,
    load_kb_profiles,
)
from scripts.evaluate_pathomeood import (
    normalize_pv_folder, normalize_pw_folder, normalize_plantdoc_folder,
)


# ---------------------------------------------------------------------------
# Encoder registry (6 visual+text encoders for the importance ablation)
# ---------------------------------------------------------------------------

ENCODERS: Dict[str, Tuple[str, Optional[str]]] = {
    "bioclip":         ("hf-hub:imageomics/bioclip",        None),
    "bioclip2":        ("hf-hub:imageomics/bioclip-2",      None),
    "clip_vitb16":     ("ViT-B-16",                         "openai"),
    "siglip_vitb16":   ("hf-hub:timm/ViT-B-16-SigLIP-256",  None),
    "fgclip":          ("hf-hub:qihoo360/fg-clip-base",     None),
    "biotrove":        ("hf-hub:BGLab/BioTrove-CLIP",       None),
}


# Sentinel value for missing state at test time. Plugged into the same
# text-tower formatter so its embedding is dimensionally identical to
# real state embeddings — TabPFN just sees "another point in state-
# embedding space" instead of a special NaN.
UNKNOWN_STATE_SENTINEL = "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--captions", required=True,
                   help="path to a captions parquet/TSV (output of "
                        "build_pathomeood_captions.py)")
    p.add_argument("--encoder", required=True, choices=sorted(ENCODERS),
                   help="frozen visual+text encoder used to produce the "
                        "image_emb, caption_emb, crop_emb and state_emb columns")
    p.add_argument("--kb-root", default="artifacts/pathome_kb")
    p.add_argument("--strategy", default=None,
                   help="caption strategy for eval-set text encoding "
                        "(default: inferred from captions file name)")
    p.add_argument("--out-train", default=None,
                   help="output train .npz (default: "
                        "data/bugwood_features/<encoder>_<strategy>.npz)")
    p.add_argument("--out-eval-root", default="data/eval_features",
                   help="output dir for eval .npz files")
    p.add_argument("--eval-pv", default=None, help="PlantVillage root")
    p.add_argument("--eval-pd", default=None, help="PlantDoc/test root")
    p.add_argument("--eval-pw", default=None, help="PlantWild root")
    p.add_argument("--crop-filter", default=None,
                   help="restrict eval to one crop (e.g. Tomato)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoder loader
# ---------------------------------------------------------------------------

def load_encoder(name: str, device: Optional[str] = None):
    """Returns (model, preprocess, tokenizer, device)."""
    try:
        import torch
        import open_clip
    except ImportError as e:
        raise SystemExit(
            f"build_features needs torch + open_clip_torch installed "
            f"(import failed: {e})"
        )
    model_name, pretrained = ENCODERS[name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def encode_images(
    model, preprocess, device, paths: Sequence[Path], batch_size: int,
) -> "np.ndarray":
    import torch
    from PIL import Image
    feats: List["np.ndarray"] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x)
            if isinstance(f, tuple):
                f = f[0]
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def encode_texts(
    model, tokenizer, device, texts: Sequence[str], batch_size: int,
) -> "np.ndarray":
    import torch
    feats: List["np.ndarray"] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            tokens = tokenizer(batch_texts).to(device)
            f = model.encode_text(tokens)
            if isinstance(f, tuple):
                f = f[0]
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)


def encode_texts_cached(
    model, tokenizer, device, texts: Sequence[str], batch_size: int,
) -> "np.ndarray":
    """Encode each UNIQUE text once and broadcast back. Crops and states
    have very few unique values; caching avoids re-tokenizing them."""
    uniq = list(dict.fromkeys(texts))
    if not uniq:
        return np.zeros((0, 512), dtype=np.float32)
    enc = encode_texts(model, tokenizer, device, uniq, batch_size)
    idx = {t: i for i, t in enumerate(uniq)}
    return np.stack([enc[idx[t]] for t in texts], axis=0)


# ---------------------------------------------------------------------------
# Text formatters for metadata embeddings
# ---------------------------------------------------------------------------

def crop_prompt(crop: str) -> str:
    return f"a photograph of a {crop} plant."


def state_prompt(state: Optional[str]) -> str:
    if not state or state.lower() in ("", "unknown", "n/a", "na"):
        return "a photograph taken at an unknown location."
    return f"a photograph taken in {state}."


# ---------------------------------------------------------------------------
# Captions I/O
# ---------------------------------------------------------------------------

def _read_caption_rows(path: Path) -> List[Dict[str, str]]:
    suf = path.suffix.lower()
    if suf == ".parquet":
        import pyarrow.parquet as pq  # type: ignore
        return pq.read_table(path).to_pylist()
    delim = "\t" if suf == ".tsv" else ","
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def _resolve_strategy(captions_path: Path, override: Optional[str]) -> str:
    if override:
        return override
    name = captions_path.stem
    if "_" in name:
        return name.split("_", 1)[1]
    return "canonical_full"


def _build_class_universe(rows: List[Dict[str, str]]) -> List[str]:
    seen = []
    for r in rows:
        label = f"{r['crop']} {r['disease']}"
        if label not in seen:
            seen.append(label)
    return seen


def _build_crop_vocab(rows: List[Dict[str, str]]) -> List[str]:
    seen = []
    for r in rows:
        if r["crop"] not in seen:
            seen.append(r["crop"])
    return seen


# ---------------------------------------------------------------------------
# Build train features
# ---------------------------------------------------------------------------

def build_train_features(
    captions_path: Path,
    encoder_name: str,
    out_path: Path,
    batch_size: int,
    device: Optional[str],
) -> Dict:
    rows = _read_caption_rows(captions_path)
    rows = [r for r in rows if (r.get("split") or "train") != "holdout"]
    if not rows:
        raise SystemExit(f"no non-holdout rows in {captions_path}")

    classes = _build_class_universe(rows)
    crops_vocab = _build_crop_vocab(rows)
    class_id = {c: i for i, c in enumerate(classes)}

    print(f"  loading encoder: {encoder_name}")
    model, preprocess, tokenizer, dev = load_encoder(encoder_name, device)

    # Filter rows with missing image files.
    keep = [Path(r["image_path"]).is_file() for r in rows]
    if not all(keep):
        n_drop = sum(1 for k in keep if not k)
        print(f"  WARNING: dropping {n_drop} rows with missing image files")
    rows = [r for r, k in zip(rows, keep) if k]

    img_paths = [Path(r["image_path"]) for r in rows]
    print(f"  encoding {len(img_paths)} images")
    image_emb = encode_images(model, preprocess, dev, img_paths, batch_size)

    print(f"  encoding {len(rows)} captions")
    captions = [r["caption_text"] for r in rows]
    caption_emb = encode_texts(model, tokenizer, dev, captions, batch_size)

    print(f"  encoding crop names (text-embedded, not one-hot)")
    crop_emb = encode_texts_cached(
        model, tokenizer, dev,
        [crop_prompt(r["crop"]) for r in rows],
        batch_size,
    )

    print(f"  encoding state metadata")
    state_emb = encode_texts_cached(
        model, tokenizer, dev,
        [state_prompt(r.get("state")) for r in rows],
        batch_size,
    )

    X = np.concatenate([image_emb, caption_emb, crop_emb, state_emb],
                       axis=1).astype(np.float32)
    y = np.array(
        [class_id[f"{r['crop']} {r['disease']}"] for r in rows],
        dtype=np.int64,
    )
    used_kb = np.array(
        [int(str(r.get("used_kb", "0")) == "1") for r in rows],
        dtype=np.int8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    block_widths = dict(
        image=int(image_emb.shape[1]),
        caption=int(caption_emb.shape[1]),
        crop=int(crop_emb.shape[1]),
        state=int(state_emb.shape[1]),
    )
    meta = dict(
        encoder=encoder_name,
        block_widths=block_widths,
        n_classes=len(classes),
        crops=crops_vocab,
    )
    np.savez_compressed(
        out_path,
        X=X, y=y, used_kb=used_kb,
        class_names=np.array(classes, dtype=object),
        meta=json.dumps(meta),
    )
    n_kb = int((used_kb == 1).sum())
    print(f"  wrote {out_path}  X={X.shape}  y={y.shape}  C={len(classes)}  "
          f"used_kb={n_kb} ({100 * n_kb / max(1, len(y)):.1f}%)")
    return meta


# ---------------------------------------------------------------------------
# Build eval features
# ---------------------------------------------------------------------------

_NORMALIZERS = {
    "plantvillage": normalize_pv_folder,
    "plantdoc":     normalize_plantdoc_folder,
    "plantwild":    normalize_pw_folder,
}


def _collect_eval_paths(root: Path, kind: str, crop_filter: Optional[str]):
    norm = _NORMALIZERS[kind]
    paths: List[Path] = []
    labels: List[Tuple[str, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        parsed = norm(sub.name)
        if parsed is None:
            continue
        folder_crop, folder_disease = parsed
        if crop_filter and folder_crop.lower() != crop_filter.lower():
            continue
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            files.extend(sub.glob(f"*{ext}"))
        files = sorted(files)
        for f in files:
            paths.append(f)
            labels.append((folder_crop, folder_disease))
    return paths, labels


def build_eval_features(
    eval_root: Path,
    eval_kind: str,
    encoder_name: str,
    strategy: str,
    train_meta: Dict,
    train_classes: List[str],
    kb_root: Path,
    out_path: Path,
    batch_size: int,
    device: Optional[str],
    crop_filter: Optional[str] = None,
) -> None:
    paths, labels = _collect_eval_paths(eval_root, eval_kind, crop_filter)
    if not paths:
        print(f"  [{eval_kind}] no images for crop={crop_filter}")
        return
    print(f"  [{eval_kind}] {len(paths)} images")

    model, preprocess, tokenizer, dev = load_encoder(encoder_name, device)

    image_emb = encode_images(model, preprocess, dev, paths, batch_size)

    # Caption embedding via KB lookup per (crop, disease).
    profiles = load_kb_profiles(str(kb_root))
    captions: List[str] = []
    used_kb_eval: List[int] = []
    for crop, disease in labels:
        if disease.lower() == "healthy":
            captions.append(build_healthy_caption(crop))
            used_kb_eval.append(0)
            continue
        rec = profiles.get((crop, disease))
        if rec is None:
            captions.append(build_fallback_caption(crop, disease, strategy))
            used_kb_eval.append(0)
        else:
            try:
                captions.append(build_disease_caption(
                    crop=crop, disease=disease,
                    disease_record=rec, strategy=strategy, state=None,
                ))
                used_kb_eval.append(1)
            except ValueError:
                captions.append(build_fallback_caption(crop, disease, strategy))
                used_kb_eval.append(0)
    caption_emb = encode_texts(model, tokenizer, dev, captions, batch_size)

    crop_emb = encode_texts_cached(
        model, tokenizer, dev,
        [crop_prompt(c) for c, _ in labels],
        batch_size,
    )

    # At test time we don't know the state. Use the sentinel embedding;
    # TabPFN will treat it as a fixed point in state-space.
    state_emb = encode_texts_cached(
        model, tokenizer, dev,
        [state_prompt(UNKNOWN_STATE_SENTINEL)] * len(labels),
        batch_size,
    )

    X = np.concatenate([image_emb, caption_emb, crop_emb, state_emb],
                       axis=1).astype(np.float32)

    class_to_id = {c: i for i, c in enumerate(train_classes)}
    y = np.array([
        class_to_id.get(f"{crop} {disease}", -1)
        for crop, disease in labels
    ], dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, y=y,
        used_kb=np.array(used_kb_eval, dtype=np.int8),
        class_names=np.array(train_classes, dtype=object),
        eval_paths=np.array([str(p) for p in paths], dtype=object),
        eval_labels=np.array(labels, dtype=object),
    )
    n_in_train = (y >= 0).sum()
    print(f"  wrote {out_path}  X={X.shape}  y={y.shape}  "
          f"in-train-class={int(n_in_train)} oof={int(len(y) - n_in_train)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    captions_path = Path(args.captions)
    strategy = _resolve_strategy(captions_path, args.strategy)
    encoder_tag = args.encoder

    out_train = Path(args.out_train) if args.out_train else (
        Path("data/bugwood_features") / f"{encoder_tag}_{strategy}.npz"
    )
    print(f"=== build_features (train) ===")
    print(f"  captions     : {captions_path}")
    print(f"  encoder      : {encoder_tag}")
    print(f"  strategy     : {strategy}")
    print(f"  out_train    : {out_train}")
    meta = build_train_features(
        captions_path, encoder_tag, out_train, args.batch_size, args.device,
    )

    npz = np.load(out_train, allow_pickle=True)
    train_classes = npz["class_names"].tolist()

    eval_root = Path(args.out_eval_root)
    for kind, src in (
        ("plantvillage", args.eval_pv),
        ("plantdoc",     args.eval_pd),
        ("plantwild",    args.eval_pw),
    ):
        if not src:
            continue
        root = Path(src)
        if not root.is_dir():
            print(f"  [{kind}] root not found: {root} — skipping")
            continue
        out_eval = eval_root / f"{encoder_tag}_{strategy}_{kind}.npz"
        print(f"\n=== build_features (eval: {kind}) ===")
        print(f"  root         : {root}")
        print(f"  out          : {out_eval}")
        build_eval_features(
            root, kind, encoder_tag, strategy, meta, train_classes,
            Path(args.kb_root), out_eval, args.batch_size, args.device,
            crop_filter=args.crop_filter,
        )


if __name__ == "__main__":
    main()
