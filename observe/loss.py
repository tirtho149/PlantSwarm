"""
observe/loss.py
===============
Multi-task uncertainty-calibration loss for OBSERVE.

Routing / backtrack heads are gone (no Algorithm 1 routing), so the
objective is now just three components::

    L = 0.4 * L_cal + 0.2 * L_cons + 0.3 * L_OC

L_cal   MSE on sigmoid-activated epsilon / alpha / c vs targets
L_cons  uncertainty-budget regularizer:
            | sigmoid(eps) + sigmoid(alpha) - (1 - sigmoid(c)) |
L_OC    BCE-with-logits for overconfidence flag

This module expects RAW LOGITS for every head; activations are applied
inside the loss for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObserveLossWeights:
    calibration:    float = 0.4
    consistency:    float = 0.2
    overconfidence: float = 0.3


@dataclass
class ObserveLossOutputs:
    total:          torch.Tensor
    calibration:    torch.Tensor
    consistency:    torch.Tensor
    overconfidence: torch.Tensor


def calibration_loss(
    epsilon_logit:    torch.Tensor, epsilon_target:    torch.Tensor,
    aleatoric_logit:  torch.Tensor, aleatoric_target:  torch.Tensor,
    confidence_logit: torch.Tensor, confidence_target: torch.Tensor,
) -> torch.Tensor:
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
    eps = torch.sigmoid(epsilon_logit)
    alp = torch.sigmoid(aleatoric_logit)
    c   = torch.sigmoid(confidence_logit)
    return (eps + alp - (1.0 - c)).abs().mean()


def overconfidence_loss(oc_logit: torch.Tensor, oc_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(oc_logit, oc_target)


class ObserveLoss(nn.Module):
    def __init__(self, weights: Optional[ObserveLossWeights] = None):
        super().__init__()
        self.w = weights or ObserveLossWeights()

    def forward(
        self,
        *,
        epsilon_logit:     torch.Tensor, epsilon_target:    torch.Tensor,
        aleatoric_logit:   torch.Tensor, aleatoric_target:  torch.Tensor,
        confidence_logit:  torch.Tensor, confidence_target: torch.Tensor,
        oc_logit:          torch.Tensor, oc_target:         torch.Tensor,
    ) -> ObserveLossOutputs:
        l_cal  = calibration_loss(
            epsilon_logit, epsilon_target,
            aleatoric_logit, aleatoric_target,
            confidence_logit, confidence_target,
        )
        l_cons = consistency_loss(epsilon_logit, aleatoric_logit, confidence_logit)
        l_oc   = overconfidence_loss(oc_logit, oc_target)
        total = (
            self.w.calibration   * l_cal
            + self.w.consistency * l_cons
            + self.w.overconfidence * l_oc
        )
        return ObserveLossOutputs(
            total=total, calibration=l_cal,
            consistency=l_cons, overconfidence=l_oc,
        )
