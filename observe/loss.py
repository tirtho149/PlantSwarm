"""
observe/loss.py
===============
Contrastive losses for OBSERVE.

Two flavours:
  - softmax_classification_loss : standard cross-entropy over cosine
                                  logits (works when each image has
                                  exactly one positive class).
  - sigmoid_pairwise_loss       : SigLIP-style sigmoid loss over all
                                  (image, class) pairs (each pair is
                                  binary: positive iff label matches).

Both expect RAW logits (already scaled by the model's logit_scale).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_classification_loss(
    logits: torch.Tensor,   # [B, C]
    targets: torch.Tensor,  # [B] int
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Standard CE over cosine·temperature logits."""
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def sigmoid_pairwise_loss(
    logits: torch.Tensor,    # [B, C]
    targets: torch.Tensor,   # [B] int
) -> torch.Tensor:
    """SigLIP-style sigmoid loss: BCE over all (image, class) pairs."""
    one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
    return F.binary_cross_entropy_with_logits(logits, one_hot)


def supervised_contrastive_loss(
    image_embeds: torch.Tensor,        # [B, D]
    text_proto_embeds: torch.Tensor,   # [C, D]
    targets: torch.Tensor,             # [B] int
    logit_scale: torch.Tensor,         # scalar
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """End-to-end variant — combines forward + softmax CE."""
    logits = logit_scale.exp() * image_embeds @ text_proto_embeds.t()
    return softmax_classification_loss(logits, targets, label_smoothing=label_smoothing)
