"""
observe/loss.py
===============
Multi-task losses for OBSERVE training (paper §7.3).

Full objective (paper §7.3, eq. L):

    L = L_rt + 0.4 * L_cal + 0.2 * L_cons + 0.2 * L_bel + 0.3 * L_OC

Components:
  L_rt   — routing cross-entropy (5-class softmax)
  L_cal  — calibration MSE on (epsilon, aleatoric, confidence)
  L_cons — uncertainty-budget regularizer (NOVEL, paper §7.1):
              L_cons = | epsilon_t + alpha_t - (1 - c_t) |
           High c_t forces both uncertainties low; low c_t requires at
           least one elevated, preventing degenerate solutions.
  L_bel  — belief-state language-model loss (token CE on belief s_t)
  L_OC   — overconfidence binary cross-entropy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObserveLossWeights:
    routing: float = 1.0
    calibration: float = 0.4
    consistency: float = 0.2
    belief: float = 0.2
    overconfidence: float = 0.3


@dataclass
class ObserveLossOutputs:
    total: torch.Tensor
    routing: torch.Tensor
    calibration: torch.Tensor
    consistency: torch.Tensor
    belief: torch.Tensor
    overconfidence: torch.Tensor


def consistency_loss(
    epsilon: torch.Tensor,
    aleatoric: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """
    Uncertainty-budget regularizer L_cons (paper §7.1).
        | eps + alpha - (1 - c) |
    """
    return (epsilon + aleatoric - (1.0 - confidence)).abs().mean()


def calibration_loss(
    epsilon_pred: torch.Tensor, epsilon_target: torch.Tensor,
    aleatoric_pred: torch.Tensor, aleatoric_target: torch.Tensor,
    confidence_pred: torch.Tensor, confidence_target: torch.Tensor,
) -> torch.Tensor:
    """MSE over the three uncertainty scalars."""
    return (
        F.mse_loss(epsilon_pred, epsilon_target)
        + F.mse_loss(aleatoric_pred, aleatoric_target)
        + F.mse_loss(confidence_pred, confidence_target)
    ) / 3.0


def routing_loss(routing_logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(routing_logits, target_class)


def overconfidence_loss(oc_pred: torch.Tensor, oc_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(oc_pred, oc_target)


def belief_loss(
    logits: torch.Tensor,           # [B, T, V]
    target_ids: torch.Tensor,       # [B, T]
    pad_id: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_id,
    )


class ObserveLoss(nn.Module):
    """Composes the full multi-task loss with paper-default weights."""

    def __init__(self, weights: Optional[ObserveLossWeights] = None):
        super().__init__()
        self.w = weights or ObserveLossWeights()

    def forward(
        self,
        routing_logits: torch.Tensor,
        target_class: torch.Tensor,
        epsilon_pred: torch.Tensor, epsilon_target: torch.Tensor,
        aleatoric_pred: torch.Tensor, aleatoric_target: torch.Tensor,
        confidence_pred: torch.Tensor, confidence_target: torch.Tensor,
        oc_pred: torch.Tensor, oc_target: torch.Tensor,
        belief_logits: Optional[torch.Tensor] = None,
        belief_targets: Optional[torch.Tensor] = None,
    ) -> ObserveLossOutputs:
        l_rt = routing_loss(routing_logits, target_class)
        l_cal = calibration_loss(
            epsilon_pred, epsilon_target,
            aleatoric_pred, aleatoric_target,
            confidence_pred, confidence_target,
        )
        l_cons = consistency_loss(epsilon_pred, aleatoric_pred, confidence_pred)
        l_oc = overconfidence_loss(oc_pred, oc_target)
        if belief_logits is not None and belief_targets is not None:
            l_bel = belief_loss(belief_logits, belief_targets)
        else:
            l_bel = torch.zeros((), device=routing_logits.device)

        total = (
            self.w.routing * l_rt
            + self.w.calibration * l_cal
            + self.w.consistency * l_cons
            + self.w.belief * l_bel
            + self.w.overconfidence * l_oc
        )
        return ObserveLossOutputs(
            total=total,
            routing=l_rt,
            calibration=l_cal,
            consistency=l_cons,
            belief=l_bel,
            overconfidence=l_oc,
        )
