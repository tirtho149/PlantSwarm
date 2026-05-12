"""
observe/grpo.py
================
Phase B trainer: Group-Relative Policy Optimization (paper §7.3), ported
to the delta-mode reward signal.

Reward function (delta mode)::

    r(trace) = routing_acc * (1 - kappa_ece) - lambda_len * len(path) / Tmax

GRPO: for each (image, context) "prompt", sample K candidate actions
from the current policy. Compute the advantage A_k = r_k - mean(r) and
clip-ratio update against the Phase-A reference policy:

    L_PPO(θ) = E_t[ min( ratio * A,
                         clip(ratio, 1 - ε, 1 + ε) * A ) ]
    L_KL     = β * KL(π_θ || π_ref)
    L_total  = -L_PPO + L_KL

Hyperparameters (paper Appendix C):
    AdamW lr=5e-5, K=8 rollouts/instance, eps_clip=0.2, beta_KL=0.04,
    epochs=10, 1x A100 40GB, ~6-8 h.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .decision_transformer import terminal_reward_for_trace
from .model import OBSERVE
from .trainer import (
    OBSERVETrainer,
    load_phase0r_traces,
    split_annotations,
)


@dataclass
class GRPOConfig:
    lr:                    float = 5e-5
    epochs:                int   = 10
    rollouts_per_instance: int   = 8
    clip_eps:              float = 0.2
    beta_kl:               float = 0.04
    lambda_len:            float = 0.05    # path-length penalty weight
    batch_size:            int   = 4
    Tmax:                  int   = 15


class GRPOTrainer:
    """GRPO refinement on top of a Phase-A (DT) checkpoint.

    Each prompt produces ``rollouts_per_instance`` sampled actions.
    Advantages are computed within the group, fed into a clipped-ratio
    policy gradient against the reference (Phase-A) policy, KL-anchored.
    """

    def __init__(
        self,
        policy: OBSERVE,
        reference: Optional[OBSERVE] = None,
        cfg: Optional[GRPOConfig] = None,
    ):
        self.policy    = policy
        self.reference = reference or copy.deepcopy(policy).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.cfg = cfg or GRPOConfig()
        self.bc = OBSERVETrainer(policy, lr=self.cfg.lr)
        self.optimizer = self.bc.optimizer

    @staticmethod
    def trace_reward(record: dict, cfg: GRPOConfig) -> float:
        base = terminal_reward_for_trace(record)
        path_len = len(record.get("path") or [])
        length_penalty = cfg.lambda_len * (path_len / max(cfg.Tmax, 1))
        return float(base - length_penalty)

    def grpo_step(self, batch: dict, device: torch.device) -> dict:
        """One GRPO update on a batch.

        We sample K candidate routing actions per row, score each by
        whether it matches the swarm's observed transition, compute the
        within-group advantage A = r - mean(r), and apply the clipped
        ratio update against the reference policy.
        """
        self.policy.train()
        self.optimizer.zero_grad()

        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in batch["inputs"].items()}
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        out = self.policy(inputs)
        with torch.no_grad():
            ref_out = self.reference(inputs)

        policy_logp = F.log_softmax(out["routing_logits"], dim=-1)
        ref_logp    = F.log_softmax(ref_out["routing_logits"], dim=-1)
        target      = labels["next_agent"]

        b = target.size(0)
        sampled = torch.multinomial(
            policy_logp.exp(),
            num_samples=self.cfg.rollouts_per_instance,
            replacement=True,
        )                                                        # [B, K]
        rewards = (sampled == target.unsqueeze(-1)).float()       # [B, K]
        adv     = rewards - rewards.mean(dim=-1, keepdim=True)

        idx_b   = torch.arange(b, device=device).unsqueeze(-1).expand_as(sampled)
        logp_s  = policy_logp[idx_b, sampled]
        refp_s  = ref_logp[idx_b,    sampled]
        ratio   = (logp_s - refp_s).exp()
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
        l_ppo   = -torch.min(ratio * adv, clipped * adv).mean()
        kl      = (policy_logp.exp() * (policy_logp - ref_logp)).sum(dim=-1).mean()
        total   = l_ppo + self.cfg.beta_kl * kl

        total.backward()
        self.optimizer.step()
        return {
            "total":  float(total.item()),
            "ppo":    float(l_ppo.item()),
            "kl":     float(kl.item()),
            "mean_r": float(rewards.mean().item()),
        }

    def fit(
        self,
        traces_path: str | Path,
        save_dir: str | Path,
        device: torch.device,
    ) -> dict:
        annotations = load_phase0r_traces(traces_path)
        if not annotations:
            raise RuntimeError(f"no annotations from {traces_path}")
        splits = split_annotations(annotations, val_frac=0.1, held_frac=0.1)
        loader = self.bc.make_loader(splits["train"], self.cfg.batch_size, shuffle=True)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            sums = {"total": 0.0, "ppo": 0.0, "kl": 0.0, "mean_r": 0.0}
            n = 0
            for batch in loader:
                stats = self.grpo_step(batch, device)
                for k in sums:
                    sums[k] += stats[k]
                n += 1
            avg = {k: v / max(n, 1) for k, v in sums.items()}
            history.append({"epoch": epoch, **avg})
            print(f"[GRPO epoch {epoch:2d}]  total={avg['total']:.4f}  "
                  f"ppo={avg['ppo']:.4f}  kl={avg['kl']:.4f}  r={avg['mean_r']:.3f}")
            self.bc.save(save_dir / f"observe_grpo_epoch_{epoch}.pt")
        self.bc.save(save_dir / "observe_grpo_last.pt")
        (save_dir / "grpo_history.json").write_text(json.dumps(history, indent=2))
        return {"history": history}
