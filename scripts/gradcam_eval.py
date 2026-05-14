"""
scripts/gradcam_eval.py
=======================
Grad-CAM visualization for the PathomeOOD frozen-encoder pipeline.
Reproduces the qualitative analysis from BioCAP paper §C.3 and the
quantitative localization score from Table 14.

Methodology (matches the paper)
-------------------------------
For a CLIP-style model:

    V = f_img(I)              image embedding (visual encoder)
    T = f_text(prompt)        text embedding   (paired text encoder)

The cosine-similarity logit is:

    s = (V . T) / (||V|| ||T||)

We backpropagate s through the final ViT transformer block to obtain
patch-token gradients. Grad-CAM = ReLU(mean_grad . feature_map),
reshaped to the patch grid (e.g. 14×14 for ViT-B/16 at 224 input),
upsampled to 224×224, overlaid on the original image.

Quantitative (Table 14 analog)
------------------------------
If the eval set ships bounding boxes (e.g. CUB-style annotations),
we compute the energy-based pointing-game score per [Wang et al.,
2020]: the fraction of Grad-CAM energy that lands inside the GT box.
Reported per encoder × dataset.

Outputs
-------
  results/figures/gradcam/<encoder>/<dataset>/<class>/<image>.png
      PNG: original image + heatmap + overlay
  results/pathomeood_eval/<encoder>_gradcam/<dataset>.json
      JSON: {n_images, mean_pointing_score, per_class}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.build_features import ENCODERS
from scripts.evaluate_pathomeood import (
    normalize_pv_folder, normalize_pw_folder, normalize_plantdoc_folder,
)


_NORMALIZERS = {
    "plantvillage": normalize_pv_folder,
    "plantdoc":     normalize_plantdoc_folder,
    "plantwild":    normalize_pw_folder,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    p.add_argument("--eval-root", required=True,
                   help="eval set root (folder-per-class)")
    p.add_argument("--eval-kind", required=True,
                   choices=sorted(_NORMALIZERS),
                   help="how to normalize folder names → (crop, disease)")
    p.add_argument("--out-fig-root", default="results/figures/gradcam")
    p.add_argument("--out-json-root", default="results/pathomeood_eval")
    p.add_argument("--max-per-class", type=int, default=3,
                   help="number of Grad-CAM heatmaps to save per class")
    p.add_argument("--crop-filter", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-csv", default=None,
                   help="optional CSV with bounding boxes (image_path, x, y, w, h) "
                        "for the energy-based pointing-game score")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoder + Grad-CAM hooks
# ---------------------------------------------------------------------------

def _load_encoder_for_gradcam(encoder: str, device: Optional[str] = None):
    """Load encoder, register forward + backward hooks on the final
    transformer block so we can read activations + gradients."""
    try:
        import torch
        import open_clip
    except ImportError as e:
        raise SystemExit(f"torch + open_clip required: {e}")

    model_name, pretrained = ENCODERS[encoder]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Find the final visual transformer block. open_clip's visual is
    # usually `model.visual.transformer.resblocks[-1]` for ViT-based
    # encoders. Fall back to last named module if not found.
    target_module = None
    try:
        target_module = model.visual.transformer.resblocks[-1]
    except (AttributeError, IndexError):
        for name, m in model.visual.named_modules():
            if "resblocks" in name or "blocks" in name:
                target_module = m
    if target_module is None:
        raise RuntimeError(f"could not locate final transformer block "
                           f"in {encoder}'s visual encoder")

    activations: Dict = {}
    gradients: Dict = {}

    def _fwd(_module, _inp, out):
        # out is (B, N_tokens, D)
        activations["v"] = out

    def _bwd(_module, _grad_in, grad_out):
        gradients["v"] = grad_out[0]

    fwd_handle = target_module.register_forward_hook(_fwd)
    bwd_handle = target_module.register_full_backward_hook(_bwd)
    return dict(
        model=model, preprocess=preprocess, tokenizer=tokenizer,
        device=device, activations=activations, gradients=gradients,
        fwd_handle=fwd_handle, bwd_handle=bwd_handle,
    )


def _compute_gradcam(ctx: Dict, image_path: Path, class_name: str):
    import torch
    from PIL import Image

    model = ctx["model"]; preprocess = ctx["preprocess"]
    tokenizer = ctx["tokenizer"]; device = ctx["device"]
    activations = ctx["activations"]; gradients = ctx["gradients"]

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device).requires_grad_(False)

    # Tokenize + encode class-name template.
    text_ids = tokenizer([f"a photo of {class_name}."]).to(device)

    activations.clear(); gradients.clear()

    # Forward: image features (drives the visual hook).
    image_features = model.encode_image(x)
    if isinstance(image_features, tuple):
        image_features = image_features[0]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        text_features = model.encode_text(text_ids)
        if isinstance(text_features, tuple):
            text_features = text_features[0]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine logit s = V . T (already unit-normalized).
    logit = (image_features * text_features).sum(dim=-1)
    model.zero_grad(set_to_none=True)
    logit.backward()

    A = activations.get("v")    # (1, N+1, D)
    G = gradients.get("v")      # (1, N+1, D)
    if A is None or G is None:
        return None, 0.0
    # Drop CLS token (index 0) — only patch tokens form the spatial grid.
    A = A[:, 1:, :]
    G = G[:, 1:, :]
    # Channel-wise mean gradient = weight per feature channel.
    weights = G.mean(dim=1, keepdim=True)      # (1, 1, D)
    cam = (A * weights).sum(dim=-1).squeeze(0) # (N,)
    cam = torch.relu(cam)
    n_tokens = cam.shape[0]
    side = int(n_tokens ** 0.5)
    if side * side != n_tokens:
        return None, 0.0
    cam = cam.view(side, side).detach().cpu().numpy()
    cam_max = cam.max() if cam.max() > 0 else 1.0
    cam = cam / cam_max  # [0, 1]
    return cam, float(logit.detach().cpu().item())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_heatmap(out_path: Path, image_path: Path, cam):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), dpi=110)
    axes[0].imshow(img); axes[0].set_title("input", fontsize=9); axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM", fontsize=9); axes[1].axis("off")
    # Overlay
    import numpy as _np
    cam_u8 = (cam * 255).astype("uint8")
    cam_img = Image.fromarray(cam_u8).resize((224, 224), Image.BILINEAR)
    cam_arr = _np.array(cam_img, dtype=_np.float32) / 255.0
    overlay = (0.5 * _np.array(img, dtype=_np.float32) / 255.0
               + 0.5 * plt.cm.jet(cam_arr)[..., :3])
    overlay = (overlay * 255).clip(0, 255).astype("uint8")
    axes[2].imshow(overlay); axes[2].set_title("overlay", fontsize=9); axes[2].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Energy-pointing-game (optional, if bboxes available)
# ---------------------------------------------------------------------------

def _energy_pointing(cam, bbox_xywh_norm) -> float:
    """Fraction of CAM energy inside the bbox. Bbox is (x, y, w, h) in
    [0, 1] normalized coordinates."""
    import numpy as _np
    H, W = cam.shape
    x0 = int(round(bbox_xywh_norm[0] * W))
    y0 = int(round(bbox_xywh_norm[1] * H))
    x1 = int(round((bbox_xywh_norm[0] + bbox_xywh_norm[2]) * W))
    y1 = int(round((bbox_xywh_norm[1] + bbox_xywh_norm[3]) * H))
    x0 = max(0, min(W - 1, x0)); x1 = max(x0 + 1, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(y0 + 1, min(H, y1))
    total = cam.sum() + 1e-12
    inside = cam[y0:y1, x0:x1].sum()
    return float(inside / total)


def _load_bboxes(path: Optional[Path]) -> Dict[str, Tuple[float, float, float, float]]:
    if not path or not path.is_file():
        return {}
    import csv as _csv
    out: Dict[str, Tuple[float, float, float, float]] = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            ip = row.get("image_path") or row.get("path") or row.get("file")
            if not ip:
                continue
            try:
                out[ip] = (
                    float(row["x"]), float(row["y"]),
                    float(row["w"]), float(row["h"]),
                )
            except (KeyError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _collect_eval_paths(root: Path, kind: str, crop_filter: Optional[str]):
    norm = _NORMALIZERS[kind]
    out_per_class: Dict[str, List[Path]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        parsed = norm(sub.name)
        if parsed is None:
            continue
        folder_crop, folder_disease = parsed
        if crop_filter and folder_crop.lower() != crop_filter.lower():
            continue
        label = f"{folder_crop} {folder_disease}"
        files = []
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            files.extend(sub.glob(f"*{ext}"))
        out_per_class[label] = sorted(files)
    return out_per_class


def main() -> None:
    args = parse_args()
    root = Path(args.eval_root)
    if not root.is_dir():
        raise SystemExit(f"eval root not found: {root}")

    per_class = _collect_eval_paths(root, args.eval_kind, args.crop_filter)
    if not per_class:
        raise SystemExit(f"no images matched under {root}")
    print(f"=== gradcam_eval ===")
    print(f"  encoder    : {args.encoder}")
    print(f"  eval set   : {args.eval_root} ({args.eval_kind})")
    print(f"  classes    : {len(per_class)}")
    print(f"  per class  : {args.max_per_class} heatmaps")

    ctx = _load_encoder_for_gradcam(args.encoder, args.device)
    bboxes = _load_bboxes(Path(args.bbox_csv) if args.bbox_csv else None)

    fig_root = Path(args.out_fig_root) / args.encoder / args.eval_kind
    summary = {"encoder": args.encoder, "eval_kind": args.eval_kind,
               "per_class": {}, "n_heatmaps": 0,
               "mean_pointing_score": None}
    all_scores: List[float] = []
    for class_name, paths in per_class.items():
        keep = paths[: args.max_per_class]
        per_class_scores: List[float] = []
        for p in keep:
            try:
                cam, logit = _compute_gradcam(ctx, p, class_name)
            except Exception as e:
                print(f"  [{class_name}] {p.name}: error {type(e).__name__}: {e}")
                continue
            if cam is None:
                continue
            out_png = (fig_root / class_name.replace("/", "_") /
                       (p.stem + ".png"))
            _save_heatmap(out_png, p, cam)
            summary["n_heatmaps"] += 1
            if str(p) in bboxes:
                per_class_scores.append(_energy_pointing(cam, bboxes[str(p)]))
        if per_class_scores:
            summary["per_class"][class_name] = float(np.mean(per_class_scores))
            all_scores.extend(per_class_scores)
    if all_scores:
        summary["mean_pointing_score"] = float(np.mean(all_scores))
        summary["n_with_bbox"] = len(all_scores)
        print(f"  energy pointing-game mean = {summary['mean_pointing_score']:.3f} "
              f"(n_with_bbox={len(all_scores)})")

    out_dir = Path(args.out_json_root) / f"{args.encoder}_gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.eval_kind}.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_dir / (args.eval_kind + '.json')}")
    print(f"  heatmaps under {fig_root}")

    ctx["fwd_handle"].remove()
    ctx["bwd_handle"].remove()


if __name__ == "__main__":
    main()
