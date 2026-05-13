"""
observe/model.py
================
OBSERVE — KB-augmented OOD plant-disease classifier.

Architecture
------------
- SigLIP-2 vision tower (frozen)  +  SigLIP-2 text tower (frozen)
- LoRA adapters on vision q/k/v   (only ~5M trainable on a ~400M base)
- No class head — classification = cosine sim against per-class text
  prototypes built from canonical + regional KB blocks (PathomeDB).
- Learnable logit-scale temperature (standard CLIP/SigLIP setting).

Why this for Bugwood -> PlantVillage / PlantWild OOD
----------------------------------------------------
The visual style gap from field photo (Bugwood) to lab cutout (PV) is
huge, but the *disease identity* is the same. A classifier conditioned
on text descriptions is invariant to visual style shifts the way a
pure visual classifier isn't. SigLIP-2 is already contrastively
pretrained on web-scale images and text; LoRA on the vision tower is
the right cheap intervention.

Open vocabulary
---------------
At test time, the model can score any disease that has a KB-derived
text prototype — including diseases the model never saw images of
during training. PV-Tomato classes (10) and PW-Tomato classes
(variable) are scored zero-shot if they aren't in the Bugwood
training set, with the KB carrying their identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassificationResult:
    """Single-image classification output."""
    class_idx: int
    class_label: str
    confidence: float                       # softmax probability of the top class
    topk: List[tuple]                       # [(idx, label, prob), ...] top-k
    logits: torch.Tensor                    # raw [C] cosine·temperature logits


class OBSERVE(nn.Module):
    """SigLIP-2 + LoRA, image-to-text-prototype classifier."""

    def __init__(
        self,
        backbone: str = "google/siglip-base-patch16-224",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        train_text_tower: bool = False,
        init_logit_scale: float = 2.6592,   # SigLIP default
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoProcessor

        self.backbone_name = backbone
        self.train_text_tower = train_text_tower

        self.processor = AutoProcessor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(
            backbone, torch_dtype=torch.bfloat16,
        )

        # LoRA on vision tower attention projections only (cheap + targets
        # the style-shift modulation). Text tower stays frozen unless
        # train_text_tower is set.
        target_modules = ["q_proj", "k_proj", "v_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora_config)
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass

        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> L2-normalised [B, D]."""
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[B, T] -> L2-normalised [B, D]. Text tower frozen by default."""
        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        feats = self.model.get_text_features(**kwargs)
        return F.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Forward — image vs precomputed class prototypes
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,        # [B, 3, H, W]
        class_proto_embeds: torch.Tensor,  # [C, D] (precomputed text embeddings)
    ) -> torch.Tensor:
        """Return scaled cosine logits [B, C]."""
        img = self.encode_image(pixel_values)
        logits = self.logit_scale.exp() * img @ class_proto_embeds.t()
        return logits

    # ------------------------------------------------------------------
    # Inference helper — top-k classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def classify(
        self,
        pixel_values: torch.Tensor,        # [B, 3, H, W] OR [3, H, W]
        class_proto_embeds: torch.Tensor,  # [C, D]
        class_labels: List[str],
        topk: int = 5,
    ) -> List[ClassificationResult]:
        """Batched (or single-image) classification."""
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        logits = self.forward(pixel_values, class_proto_embeds.to(device))
        probs  = F.softmax(logits.float(), dim=-1)
        results: List[ClassificationResult] = []
        k = min(topk, probs.size(-1))
        topv, topi = probs.topk(k, dim=-1)
        for b in range(probs.size(0)):
            idxs  = topi[b].tolist()
            probs_top = topv[b].tolist()
            results.append(ClassificationResult(
                class_idx=idxs[0],
                class_label=class_labels[idxs[0]],
                confidence=float(probs_top[0]),
                topk=[(i, class_labels[i], float(p)) for i, p in zip(idxs, probs_top)],
                logits=logits[b].detach().cpu(),
            ))
        return results
