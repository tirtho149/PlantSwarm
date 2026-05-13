"""
observe/inference.py
====================
Single-image classification wrapper.

Loads a trained OBSERVE checkpoint and the per-class prototype texts
stored alongside it, encodes the prototypes once, and answers
``classify(image, topk=5)`` calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .dataset import ClassIndex
from .model import ClassificationResult, OBSERVE
from .trainer import encode_class_prototypes


class OBSERVEInference:
    """Thin wrapper over OBSERVE for single-image inference."""

    def __init__(
        self,
        ckpt_path: str | Path,
        backbone: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        labels = ckpt.get("class_labels") or []
        prototype_texts = ckpt.get("prototype_texts") or []
        if not labels or not prototype_texts:
            raise ValueError(
                "Checkpoint missing class_labels or prototype_texts. "
                "Was it produced by observe.trainer.OBSERVETrainer.save()?"
            )

        self.class_index = ClassIndex(labels)
        self.model = OBSERVE(
            backbone=backbone, lora_r=lora_r, lora_alpha=lora_alpha,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.proto_embeds = encode_class_prototypes(
            self.model, prototype_texts, device=self.device,
        )

    def classify(self, image, topk: int = 5) -> ClassificationResult:
        """Single PIL image (or path) -> top-1 + top-k."""
        from PIL import Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        proc = self.model.processor(images=image, return_tensors="pt")
        results = self.model.classify(
            pixel_values=proc["pixel_values"][0],
            class_proto_embeds=self.proto_embeds,
            class_labels=self.class_index.labels,
            topk=topk,
        )
        return results[0]
