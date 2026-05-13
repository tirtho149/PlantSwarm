"""
observe/trainer.py
==================
Trainer for the KB-augmented OOD classifier.

Pipeline per epoch:
  1. Encode all class prototype texts ONCE (text tower is frozen) — the
     prototype embeddings are cached and reused across all images.
  2. For each minibatch of training images:
       img_embeds = vision tower (+ LoRA) on pixel_values
       logits    = logit_scale * img_embeds @ proto_embeds.T
       loss      = cross_entropy(logits, target_class_id)
  3. Validate on the val split; track top-1 accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import ClassIndex
from .loss import softmax_classification_loss


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prototype encoder helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_class_prototypes(
    model,
    prototype_texts: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 64,
) -> torch.Tensor:
    """Tokenize + encode all class prototypes; returns [C, D] L2-normed."""
    model.eval()
    processor = model.processor
    embeds = []
    for i in range(0, len(prototype_texts), batch_size):
        batch = list(prototype_texts[i:i + batch_size])
        toks = processor(
            text=batch, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_length,
        )
        toks = {k: v.to(device) if hasattr(v, "to") else v for k, v in toks.items()}
        feats = model.encode_text(
            input_ids=toks["input_ids"],
            attention_mask=toks.get("attention_mask"),
        )
        embeds.append(feats)
    return torch.cat(embeds, dim=0)


# ---------------------------------------------------------------------------
# Image collator — runs the processor on RGB PIL images
# ---------------------------------------------------------------------------

class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples: List[dict]) -> dict:
        images = [s["image"] for s in samples]
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
        proc = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": proc["pixel_values"],
            "labels":       labels,
            "paths":        [s.get("path", "") for s in samples],
        }


# ---------------------------------------------------------------------------
# Image-grouped split (so all images of one source don't leak across folds)
# ---------------------------------------------------------------------------

def split_indices(
    dataset: Dataset,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Stratified split by class label; deterministic seed."""
    import random
    by_class: Dict[int, List[int]] = {}
    for i in range(len(dataset)):
        lbl = dataset[i]["label"]
        by_class.setdefault(int(lbl), []).append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return {"train": train_idx, "val": val_idx}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OBSERVETrainer:
    """Contrastive trainer over (image, class-prototype) pairs."""

    def __init__(
        self,
        model,
        class_index: ClassIndex,
        prototype_texts: Sequence[str],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.class_index = class_index
        self.prototype_texts = list(prototype_texts)
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )
        # Will be (re-)encoded by encode_prototypes() before each epoch.
        self._proto_embeds: Optional[torch.Tensor] = None

    def encode_prototypes(self, device: torch.device) -> None:
        self._proto_embeds = encode_class_prototypes(
            self.model, self.prototype_texts, device=device,
        )

    def make_loader(
        self,
        dataset: Dataset,
        indices: List[int],
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        from torch.utils.data import Subset
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=ImageCollator(self.model.processor),
        )

    def _step(self, batch: dict, device: torch.device, train: bool) -> dict:
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        pixels = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        assert self._proto_embeds is not None, "encode_prototypes() not called"
        logits = self.model(pixels, self._proto_embeds.to(device))
        loss = softmax_classification_loss(logits, labels)
        if train:
            loss.backward()
            self.optimizer.step()
        pred = logits.argmax(dim=-1)
        top1 = float((pred == labels).float().mean().item())
        return {"loss": float(loss.item()), "top1": top1}

    def train_epoch(self, loader: DataLoader, device: torch.device) -> dict:
        self.encode_prototypes(device)
        sums = {"loss": 0.0, "top1": 0.0}
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            stats = self._step(batch, device, train=True)
            sums["loss"] += stats["loss"]
            sums["top1"] += stats["top1"]
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, device: torch.device) -> dict:
        self.encode_prototypes(device)
        sums = {"loss": 0.0, "top1": 0.0}
        n = 0
        for batch in tqdm(loader, desc="val", leave=False):
            stats = self._step(batch, device, train=False)
            sums["loss"] += stats["loss"]
            sums["top1"] += stats["top1"]
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_labels":         self.class_index.labels,
            "prototype_texts":      self.prototype_texts,
        }, path)
        logger.info("OBSERVE checkpoint -> %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("OBSERVE checkpoint loaded <- %s", path)
