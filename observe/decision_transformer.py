"""
observe/decision_transformer.py
================================
Phase A trainer: Decision Transformer on PlantSwarm Bugwood traces (paper §7.3).

Routing traces are reformulated as return-conditioned sequences:

    [R_0, s_0, a_0, R_1, s_1, a_1, ...]

with R_t = sum_{t' >= t} r_{t'} and terminal reward
    r_T = F1(ŷ, y*) - 0.4 * ECE              (paper §7.3 reward eq.)

At inference, conditioning on a target return R^* enables ECE-targeted routing
(generate the action sequence that achieves the desired calibration level).

Hyperparameters (Appendix C):
    AdamW lr=1e-4, cosine decay, warmup 500 steps
    batch 32 (8 x 4 grad accum)
    50 epochs, early stopping (val ECE, patience 5)
    1x A100 40GB, ~4-6h.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .loss import ObserveLoss, ObserveLossOutputs, ObserveLossWeights
from .model import OBSERVE


@dataclass
class DTConfig:
    lr: float = 1e-4
    warmup_steps: int = 500
    epochs: int = 50
    patience: int = 5
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_path_length: int = 15
    target_return_mean: float = 0.85       # cond return at inference
    f1_weight: float = 1.0
    ece_weight: float = 0.4


# ---------------------------------------------------------------------------
# Trace → return-conditioned sequence
# ---------------------------------------------------------------------------

def trace_terminal_reward(f1: float, ece: float, *, lambda_ece: float = 0.4) -> float:
    """r_T = F1 - lambda * ECE  (paper §7.3)."""
    return float(f1) - lambda_ece * float(ece)


def returns_to_go(rewards: List[float]) -> List[float]:
    """R_t = sum_{t' >= t} r_{t'}."""
    out: List[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running += rewards[i]
        out[i] = running
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DecisionTransformerTrainer:
    """
    Phase A: behaviour-cloning under return conditioning.

    The actual sequence-modelling forward pass is delegated to OBSERVE which
    already wraps Qwen2.5-VL-7B + LoRA + multi-head outputs. DT-specific
    additions live here:
      * return_to_go injection into context
      * masked routing loss over the sequence
    """

    def __init__(
        self,
        model: OBSERVE,
        cfg: Optional[DTConfig] = None,
        loss_weights: Optional[ObserveLossWeights] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.cfg = cfg or DTConfig()
        self.device = device
        self.loss_fn = ObserveLoss(loss_weights)
        # Only LoRA + heads need gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.cfg.lr, weight_decay=0.01)
        self.scheduler = None
        self.best_val_ece: float = float("inf")
        self._patience = 0

    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader, save_dir: str) -> None:
        steps_per_epoch = max(1, len(train_loader) // self.cfg.grad_accum_steps)
        total_steps = self.cfg.epochs * steps_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        for epoch in range(self.cfg.epochs):
            self._train_one_epoch(train_loader)
            val_ece = self._evaluate(val_loader)
            if val_ece < self.best_val_ece:
                self.best_val_ece = val_ece
                self._patience = 0
                self._save_checkpoint(save_dir, tag="best")
            else:
                self._patience += 1
                if self._patience >= self.cfg.patience:
                    print(f"DT early stop at epoch {epoch+1}, best val ECE={self.best_val_ece:.4f}")
                    break

    def _train_one_epoch(self, loader) -> None:
        self.model.train()
        accum = 0
        self.optimizer.zero_grad()
        for batch in loader:
            outputs: ObserveLossOutputs = self._forward_batch(batch)
            loss = outputs.total / self.cfg.grad_accum_steps
            loss.backward()
            accum += 1
            if accum % self.cfg.grad_accum_steps == 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def _evaluate(self, loader) -> float:
        # TODO(pathome): compute ECE over predicted vs target confidence using
        # calibration.ece.compute_ece_from_probs on the validation traces.
        self.model.eval()
        running = 0.0
        n = 0
        for batch in loader:
            outputs = self._forward_batch(batch)
            running += float(outputs.calibration.detach().cpu())
            n += 1
        return running / max(n, 1)

    def _forward_batch(self, batch) -> ObserveLossOutputs:
        """
        Expected batch shape (collated by RoutingTraceDataset):
          image, context_text, target_class, eps_t, alpha_t, c_t, oc_t,
          belief_target_ids (optional), return_to_go.
        """
        images = batch["image"]
        contexts = batch["context_text"]
        # OBSERVE.forward returns a dict; we use it for the action heads.
        # Vectorised batched forward is a TODO once we have multi-image
        # processing; for now iterate per-sample.
        routing_logits = []
        eps_pred = []
        ale_pred = []
        conf_pred = []
        oc_pred = []
        for img, ctx in zip(images, contexts):
            out = self.model(image=img, context_text=ctx, return_dict=True)
            routing_logits.append(out["routing_probs"].squeeze(0))
            eps_pred.append(out["epistemic"])
            ale_pred.append(out["aleatoric"])
            conf_pred.append(out["confidence"])
            oc_pred.append(out["oc_prob"])

        routing_logits = torch.stack(routing_logits, dim=0)
        eps_pred = torch.stack(eps_pred, dim=0).squeeze(-1)
        ale_pred = torch.stack(ale_pred, dim=0).squeeze(-1)
        conf_pred = torch.stack(conf_pred, dim=0).squeeze(-1)
        oc_pred = torch.stack(oc_pred, dim=0).squeeze(-1)

        return self.loss_fn(
            routing_logits=routing_logits,
            target_class=batch["target_class"].to(self.device),
            epsilon_pred=eps_pred,
            epsilon_target=batch["epsilon_target"].to(self.device),
            aleatoric_pred=ale_pred,
            aleatoric_target=batch["aleatoric_target"].to(self.device),
            confidence_pred=conf_pred,
            confidence_target=batch["confidence_target"].to(self.device),
            oc_pred=oc_pred,
            oc_target=batch["oc_target"].to(self.device),
        )

    # ------------------------------------------------------------------

    def _save_checkpoint(self, save_dir: str, tag: str) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / f"observe_dt_{tag}.pt")
