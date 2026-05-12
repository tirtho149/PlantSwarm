"""
observe/loss.py
===============
Multi-task losses for OBSERVE (paper §7.3, retargeted for delta mode).

Full objective::

    L = L_rt + 0.4 * L_cal + 0.2 * L_cons + 0.3 * L_OC + 0.2 * L_bel

L_rt    routing cross-entropy from RAW logits (5-class)
L_cal   MSE on sigmoid-activated scalar heads (epsilon, aleatoric, c)
        vs. their targets
L_cons  uncertainty-budget regularizer (paper §7.1):
            | epsilon + alpha - (1 - c) |
        High c forces both uncertainties low; low c requires at least
        one elevated, preventing degenerate solutions.
L_OC    BCE-with-logits for overconfidence flag.
L_bel   optional belief-text LM loss when belief tokens are present.

This module expects **raw logits** for every head — the model emits raw
logits, the loss applies the right activations internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObserveLossWeights:
    routing:        float = 1.0
    calibration:    float = 0.4
    consistency:    float = 0.2
    belief:         float = 0.2
    overconfidence: float = 0.3


@dataclass
class ObserveLossOutputs:
    total:          torch.Tensor
    routing:        torch.Tensor
    calibration:    torch.Tensor
    consistency:    torch.Tensor
    belief:         torch.Tensor
    overconfidence: torch.Tensor


def routing_loss(routing_logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over raw routing logits ``[B, n_agents]``."""
    return F.cross_entropy(routing_logits, target_class)


def calibration_loss(
    epsilon_logit:    torch.Tensor, epsilon_target:    torch.Tensor,
    aleatoric_logit:  torch.Tensor, aleatoric_target:  torch.Tensor,
    confidence_logit: torch.Tensor, confidence_target: torch.Tensor,
) -> torch.Tensor:
    """MSE over sigmoid-activated scalars vs targets in [0, 1]."""
    return (
        F.mse_loss(torch.sigmoid(epsilon_logit),    epsilon_target)
        + F.mse_loss(torch.sigmoid(aleatoric_logit), aleatoric_target)
        + F.mse_loss(torch.sigmoid(confidence_logit), confidence_target)
    ) / 3.0


def consistency_loss(
    epsilon_logit:    torch.Tensor,
    aleatoric_logit:  torch.Tensor,
    confidence_logit: torch.Tensor,
) -> torch.Tensor:
    """| sigmoid(eps) + sigmoid(alpha) - (1 - sigmoid(c)) |"""
    eps = torch.sigmoid(epsilon_logit)
    alp = torch.sigmoid(aleatoric_logit)
    c   = torch.sigmoid(confidence_logit)
    return (eps + alp - (1.0 - c)).abs().mean()


def overconfidence_loss(oc_logit: torch.Tensor, oc_target: torch.Tensor) -> torch.Tensor:
    """BCE with logits — numerically stable vs sigmoid + BCE."""
    return F.binary_cross_entropy_with_logits(oc_logit, oc_target)


def belief_loss(
    logits:      torch.Tensor,      # [B, T, V]
    target_ids:  torch.Tensor,      # [B, T]
    pad_id:      int = -100,
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
        *,
        routing_logits:    torch.Tensor,
        target_class:      torch.Tensor,
        epsilon_logit:     torch.Tensor, epsilon_target:    torch.Tensor,
        aleatoric_logit:   torch.Tensor, aleatoric_target:  torch.Tensor,
        confidence_logit:  torch.Tensor, confidence_target: torch.Tensor,
        oc_logit:          torch.Tensor, oc_target:         torch.Tensor,
        belief_logits:     Optional[torch.Tensor] = None,
        belief_targets:    Optional[torch.Tensor] = None,
    ) -> ObserveLossOutputs:
        l_rt   = routing_loss(routing_logits, target_class)
        l_cal  = calibration_loss(
            epsilon_logit, epsilon_target,
            aleatoric_logit, aleatoric_target,
            confidence_logit, confidence_target,
        )
        l_cons = consistency_loss(epsilon_logit, aleatoric_logit, confidence_logit)
        l_oc   = overconfidence_loss(oc_logit, oc_target)
        if belief_logits is not None and belief_targets is not None:
            l_bel = belief_loss(belief_logits, belief_targets)
        else:
            l_bel = torch.zeros((), device=routing_logits.device)
        total = (
            self.w.routing       * l_rt
            + self.w.calibration   * l_cal
            + self.w.consistency   * l_cons
            + self.w.belief        * l_bel
            + self.w.overconfidence * l_oc
        )
        return ObserveLossOutputs(
            total=total, routing=l_rt, calibration=l_cal,
            consistency=l_cons, belief=l_bel, overconfidence=l_oc,
        )
