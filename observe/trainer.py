"""
observe/trainer.py
==================
OBSERVE trainer on Phase 0R per-pass trace JSONL.

Schema after the Algorithm-1 removal: one line per (tuple, pass) with
the four specialist outputs (parallel), the consolidator output, the
final delta list, and the existing-KB context the agents saw. There is
no routing path, no backtrack count.

Per-pass supervision targets are derived from the consolidator output::

    target_confidence    kappa in {high, medium, low} -> {0.9, 0.6, 0.3}
    target_epistemic     hard-coded heuristic from final-deltas count
                         vs. specialist union size — proxy for "was this
                         pass already complete after the specialists, or
                         did the consolidator need to add structure?"
    target_aleatoric     1 - kappa_scalar  (low kappa = high noise)
    target_overconfidence 1 iff kappa == "high" AND len(final_deltas) == 0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


_KAPPA_TO_SCALAR = {"high": 0.9, "medium": 0.6, "low": 0.3}


# ---------------------------------------------------------------------------
# Per-pass annotation
# ---------------------------------------------------------------------------

@dataclass
class PassAnnotation:
    image_path:        str
    crop:              str
    disease:           str
    state:             str
    context_text:      str
    # Targets ↓
    confidence:        float
    epistemic:         float
    aleatoric:         float
    overconfidence:    bool
    # Diagnostic ↓
    profile_id:        str
    pass_idx:          int
    n_final_deltas:    int
    n_specialist_union: int


def _kappa_to_scalar(k: str) -> float:
    return _KAPPA_TO_SCALAR.get(str(k or "").lower(), 0.6)


def _render_context_text(
    *,
    crop: str, disease: str, state: str,
    existing_kb: List[dict], specialist_outputs: List[dict],
) -> str:
    parts: List[str] = [
        f"Crop: {crop}", f"Disease: {disease}", f"State: {state}",
    ]
    if existing_kb:
        parts.append(f"Existing KB observations ({len(existing_kb)}):")
        for d in existing_kb:
            parts.append(f"  [{d.get('field','other')}] {d.get('image_shows','')[:140]}")
    if specialist_outputs:
        parts.append("Specialist outputs (parallel):")
        for s in specialist_outputs:
            parts.append(f"  [{s.get('agent_name', '?')}] "
                         f"(k={s.get('confidence','?')})")
            for d in s.get("deltas") or []:
                parts.append(f"      delta[{d.get('field','?')}]: "
                             f"{d.get('image_shows','')[:120]}")
    return "\n".join(parts)


def annotation_from_pass(record: dict) -> Optional[PassAnnotation]:
    """Build one supervision sample from a single per-pass trace record."""
    consolidator = record.get("consolidator_output") or {}
    specialists  = record.get("specialist_outputs") or []
    final_deltas = record.get("final_deltas") or consolidator.get("deltas") or []

    kappa = (consolidator.get("confidence") or "medium").lower()
    kappa_scalar = _kappa_to_scalar(kappa)
    n_final = len(final_deltas)
    n_union = sum(len(s.get("deltas") or []) for s in specialists)

    # Epistemic proxy: how much "structure" the consolidator added (or
    # removed) relative to the specialist union — bounded to [0, 1].
    if n_union == 0:
        epistemic = 0.0
    else:
        epistemic = max(0.0, min(1.0, abs(n_union - n_final) / max(1, n_union)))
    aleatoric = 1.0 - kappa_scalar
    overconfidence = bool(kappa == "high" and n_final == 0)

    ctx_text = _render_context_text(
        crop=record.get("crop", ""),
        disease=record.get("disease", ""),
        state=record.get("state", ""),
        existing_kb=record.get("existing_kb_at_start") or [],
        specialist_outputs=specialists,
    )
    return PassAnnotation(
        image_path=record.get("image_path", ""),
        crop=record.get("crop", ""),
        disease=record.get("disease", ""),
        state=record.get("state", ""),
        context_text=ctx_text,
        confidence=kappa_scalar,
        epistemic=epistemic,
        aleatoric=aleatoric,
        overconfidence=overconfidence,
        profile_id=record.get("profile_id", ""),
        pass_idx=int(record.get("pass_idx", 0)),
        n_final_deltas=n_final,
        n_specialist_union=n_union,
    )


def load_phase0r_traces(path: str | Path) -> List[PassAnnotation]:
    annotations: List[PassAnnotation] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ann = annotation_from_pass(rec)
            if ann is not None:
                annotations.append(ann)
    return annotations


# ---------------------------------------------------------------------------
# Dataset + collator
# ---------------------------------------------------------------------------

class PassDataset(Dataset):
    def __init__(self, annotations: Sequence[PassAnnotation]):
        self.annotations = list(annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        return {
            "image_path":     ann.image_path,
            "context_text":   ann.context_text,
            "epistemic":      float(ann.epistemic),
            "aleatoric":      float(ann.aleatoric),
            "confidence":     float(ann.confidence),
            "overconfidence": float(ann.overconfidence),
        }


class ObserveCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples: List[dict]) -> dict:
        from PIL import Image
        images = [Image.open(s["image_path"]).convert("RGB") for s in samples]
        texts  = [s["context_text"] for s in samples]
        inputs = self.processor(
            images=images, text=texts,
            return_tensors="pt", padding=True, truncation=True,
        )
        labels = {
            "epistemic":      torch.tensor([s["epistemic"]      for s in samples], dtype=torch.float32),
            "aleatoric":      torch.tensor([s["aleatoric"]      for s in samples], dtype=torch.float32),
            "confidence":     torch.tensor([s["confidence"]     for s in samples], dtype=torch.float32),
            "overconfidence": torch.tensor([s["overconfidence"] for s in samples], dtype=torch.float32),
        }
        return {"inputs": dict(inputs), "labels": labels}


# ---------------------------------------------------------------------------
# Image-grouped split (no leakage across image_path)
# ---------------------------------------------------------------------------

def split_annotations(
    annotations: Sequence[PassAnnotation],
    *,
    val_frac: float = 0.1,
    held_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    import random as _random
    by_image: dict[str, List[PassAnnotation]] = {}
    for ann in annotations:
        by_image.setdefault(ann.image_path, []).append(ann)
    unique = sorted(by_image.keys())
    rng = _random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_val  = int(n * val_frac)
    n_held = int(n * held_frac)
    n_train = n - n_val - n_held
    train_ids = set(unique[:n_train])
    val_ids   = set(unique[n_train:n_train + n_val])
    held_ids  = set(unique[n_train + n_val:])

    train_a: List[PassAnnotation] = []
    val_a:   List[PassAnnotation] = []
    held_a:  List[PassAnnotation] = []
    for img_id, anns in by_image.items():
        bucket = (train_a if img_id in train_ids
                  else val_a if img_id in val_ids
                  else held_a)
        bucket.extend(anns)
    return {
        "train": train_a, "val": val_a, "held": held_a,
        "n_images_train": len(train_ids),
        "n_images_val":   len(val_ids),
        "n_images_held":  len(held_ids),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OBSERVETrainer:
    """Calibration trainer for OBSERVE on per-pass uncertainty targets."""

    def __init__(self, model, lr: float = 1e-4, weight_decay: float = 0.01):
        self.model = model
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )

    def make_loader(
        self,
        annotations: Sequence[PassAnnotation],
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        ds = PassDataset(annotations)
        collator = ObserveCollator(self.model.processor)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=collator,
        )

    def _step(self, batch: dict, loss_fn, device: torch.device, train: bool) -> dict:
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in batch["inputs"].items()}
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        out = self.model(inputs)
        losses = loss_fn(
            epsilon_logit=out["epistemic_logit"],     epsilon_target=labels["epistemic"],
            aleatoric_logit=out["aleatoric_logit"],   aleatoric_target=labels["aleatoric"],
            confidence_logit=out["confidence_logit"], confidence_target=labels["confidence"],
            oc_logit=out["oc_logit"],                 oc_target=labels["overconfidence"],
        )
        if train:
            losses.total.backward()
            self.optimizer.step()
        return {
            "total":  float(losses.total.item()),
            "cal":    float(losses.calibration.item()),
            "cons":   float(losses.consistency.item()),
            "oc":     float(losses.overconfidence.item()),
        }

    def train_epoch(self, loader: DataLoader, loss_fn, device: torch.device) -> dict:
        sums = {"total": 0.0, "cal": 0.0, "cons": 0.0, "oc": 0.0}
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            stats = self._step(batch, loss_fn, device, train=True)
            for k in sums:
                sums[k] += stats[k]
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, loss_fn, device: torch.device) -> dict:
        sums = {"total": 0.0, "cal": 0.0, "cons": 0.0, "oc": 0.0}
        n = 0
        for batch in tqdm(loader, desc="val", leave=False):
            stats = self._step(batch, loss_fn, device, train=False)
            for k in sums:
                sums[k] += stats[k]
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("OBSERVE checkpoint -> %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("OBSERVE checkpoint loaded <- %s", path)
