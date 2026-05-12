"""
observe/grpo.py
===============
Phase B trainer: Group Relative Policy Optimization (paper §7.3).

GRPO refines the Phase-A Decision Transformer policy with an online reward
signal. For each instance we sample G=8 rollouts, compute group-relative
advantage:

    Â_i = (r_i - mean(r)) / std(r)

and apply a clipped surrogate objective with KL anchored to the Phase-A
reference policy:

    L_GRPO = -E[ min(ρ_i Â_i, clip(ρ_i, 1±ε) Â_i) ] + β_KL * KL(π_θ || π_ref)

Reward (paper §7.3):
    r = F1 - 0.4·ECE + 0.3·ΔF1_BT - 0.05·(L/L_max) + 0.2·(1 - |eps_T - eps*_T|)

Hyperparameters (Appendix C):
    AdamW lr=5e-5, G=8 rollouts per instance, 10 epochs,
    β_KL=0.04, clip ε=0.2, reference policy frozen from Phase A.
    1x A100 40GB, ~3-4h.

NOTE: the rollout-collection loop calls ``run_pipeline_and_score`` which is a
hook into the actual PlantSwarm runtime. Wiring is described in the
docstring; implementation is a TODO when integrated with ``plantswarm/``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .model import OBSERVE


@dataclass
class GRPOConfig:
    lr: float = 5e-5
    epochs: int = 10
    rollouts_per_instance: int = 8
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    f1_weight: float = 1.0
    ece_weight: float = 0.4
    bt_delta_weight: float = 0.3
    length_penalty_weight: float = 0.05
    epsilon_match_weight: float = 0.2
    max_path_length: int = 15


@dataclass
class Rollout:
    """One on-policy rollout from a single training instance."""
    log_probs: torch.Tensor          # [T] log π_θ(a_t | s_t) along the rollout
    ref_log_probs: torch.Tensor      # [T] log π_ref(a_t | s_t)
    f1: float
    ece: float
    delta_f1_bt: float
    path_length: int
    epsilon_match: float
    reward: float = 0.0


def compute_reward(roll: Rollout, cfg: GRPOConfig) -> float:
    return (
        cfg.f1_weight * roll.f1
        - cfg.ece_weight * roll.ece
        + cfg.bt_delta_weight * roll.delta_f1_bt
        - cfg.length_penalty_weight * (roll.path_length / cfg.max_path_length)
        + cfg.epsilon_match_weight * roll.epsilon_match
    )


def group_relative_advantage(rewards: List[float]) -> List[float]:
    if not rewards:
        return []
    t = torch.tensor(rewards, dtype=torch.float32)
    mean = t.mean()
    std = t.std().clamp(min=1e-6)
    return ((t - mean) / std).tolist()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    Phase B GRPO trainer. Requires a callable that runs PlantSwarm under the
    current OBSERVE policy and returns a Rollout — the integration point with
    the agent runtime.

    ``rollout_fn(model, instance) -> Rollout`` is responsible for:
      1. Calling the swarm with OBSERVE-driven routing
      2. Recording per-step log π_θ and log π_ref
      3. Scoring the final prediction (F1, ECE, BT delta, eps match)
    """

    def __init__(
        self,
        model: OBSERVE,
        cfg: Optional[GRPOConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.cfg = cfg or GRPOConfig()
        self.device = device
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.cfg.lr, weight_decay=0.01)

        # Phase-A reference policy: frozen deep copy
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    # ------------------------------------------------------------------

    def fit(
        self,
        train_instances,
        rollout_fn: Callable[[OBSERVE, OBSERVE, object], Rollout],
        save_dir: str,
    ) -> None:
        for epoch in range(self.cfg.epochs):
            for instance in train_instances:
                rollouts = self._collect_rollouts(rollout_fn, instance)
                if not rollouts:
                    continue
                rewards = [compute_reward(r, self.cfg) for r in rollouts]
                advantages = group_relative_advantage(rewards)
                loss = self._grpo_loss(rollouts, advantages)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self._save_checkpoint(save_dir, tag=f"epoch_{epoch+1}")

    # ------------------------------------------------------------------

    def _collect_rollouts(self, rollout_fn, instance) -> List[Rollout]:
        rollouts: List[Rollout] = []
        for _ in range(self.cfg.rollouts_per_instance):
            try:
                r = rollout_fn(self.model, self.ref_model, instance)
                rollouts.append(r)
            except Exception:
                # TODO(pathome): log and continue; do not abort epoch on a
                # single failed rollout.
                continue
        return rollouts

    def _grpo_loss(
        self,
        rollouts: List[Rollout],
        advantages: List[float],
    ) -> torch.Tensor:
        """Clipped surrogate + KL to reference policy."""
        losses: List[torch.Tensor] = []
        for roll, adv in zip(rollouts, advantages):
            adv_t = torch.tensor(adv, device=self.device, dtype=torch.float32)
            # Importance ratio per-token; clipped surrogate.
            ratio = (roll.log_probs - roll.ref_log_probs).exp()
            unclipped = ratio * adv_t
            clipped = ratio.clamp(1 - self.cfg.clip_eps,
                                  1 + self.cfg.clip_eps) * adv_t
            surrogate = -torch.min(unclipped, clipped).mean()

            # KL(π_θ || π_ref) approximated per-token.
            kl = (roll.log_probs - roll.ref_log_probs).mean()
            losses.append(surrogate + self.cfg.beta_kl * kl)

        return torch.stack(losses).mean()

    def _save_checkpoint(self, save_dir: str, tag: str) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / f"observe_grpo_{tag}.pt")
