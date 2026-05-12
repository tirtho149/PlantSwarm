"""
observe/trainer.py
==================
OBSERVE training pipeline — delta-mode supervision.

Consumes Phase 0R trace JSONL (written by ``plantswarm.delta_pipeline``
when ``PATHOME_TRACE_DIR`` is set). Each line is one stochastic
routed trace for one (crop, disease, state, image) tuple:

    {
      "profile_id", "crop", "disease", "state",
      "primary_image_id", "image_path",
      "run_idx", "path", "decisions",
      "confidences",           ["high", "medium", ...],
      "backtrack_count", "early_terminated",
      "context_buffer": [                    one entry per agent step:
        {
          "agent_name",
          "deltas":     [{field, canonical_says, image_shows, image_quote}],
          "confidence": "high"|"medium"|"low",
          "handoff_target", "reasoning", "raw_text"
        }, ...
      ],
      "final_deltas":          [...],
      "existing_kb_at_start":  [...],
    }

For each step in a trace, we derive supervision for OBSERVE's heads:

    target_routing    : index of path[i+1] (next agent the swarm picked)
    target_backtrack  : 1 if path[i+1] == "MorphologyAgent" and
                        path[i] != "MorphologyAgent" else 0
    target_confidence : κ → scalar (high=0.9, medium=0.6, low=0.3)
    target_epistemic  : 1 - (current_step_deltas / final_deltas);
                        i.e. how much more the trace added after this step
    target_aleatoric  : 1 - kappa_scalar (low conf = high irreducible noise)
    target_oc         : 1 if (κ=high AND |final_deltas| == 0) else 0
                        (claimed high but produced nothing — overconfident)
    target_belief     : the agent's reasoning string (text)

The image is loaded from ``image_path`` on disk on the fly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


AGENT_CLASSES: Sequence[str] = (
    "MorphologyAgent", "SymptomAgent", "PathogenAgent",
    "SeverityAgent", "DiagnosisAgent",
)
_AGENT_IDX = {a: i for i, a in enumerate(AGENT_CLASSES)}

_KAPPA_TO_SCALAR = {"high": 0.9, "medium": 0.6, "low": 0.3}


# ---------------------------------------------------------------------------
# Per-step training annotation
# ---------------------------------------------------------------------------

@dataclass
class TraceStepAnnotation:
    """One supervised training sample, derived from one step of a Phase 0R trace."""

    image_path:        str       # absolute or repo-relative
    crop:              str
    disease:           str
    state:             str
    step:              int       # index within the trace
    current_agent:    str        # which of 5
    context_text:     str        # rendered (canonical_slice + existing + prior_context)
    # Targets ↓
    next_agent:        str       # 5-class
    backtrack:         bool
    confidence:        float     # κ scalar
    epistemic:         float
    aleatoric:         float
    overconfidence:    bool
    belief_state:      str
    # Diagnostic ↓
    profile_id:        str
    run_idx:           int
    n_deltas_at_step:  int
    n_deltas_final:    int


def _kappa_to_scalar(k: str) -> float:
    return _KAPPA_TO_SCALAR.get(str(k or "").lower(), 0.6)


def _render_context_text(
    *,
    crop: str,
    disease: str,
    state: str,
    existing_kb: List[dict],
    context_buffer_up_to_step: List[dict],
) -> str:
    """Cheap text-only render of the agent's prompt context for training.

    Doesn't reproduce the full DELTA_USER_PROMPT format — just enough
    structural signal for OBSERVE to imitate routing decisions.
    """
    parts: List[str] = [
        f"Crop: {crop}", f"Disease: {disease}", f"State: {state}",
    ]
    if existing_kb:
        parts.append(f"Existing KB observations ({len(existing_kb)}):")
        for d in existing_kb:
            parts.append(f"  [{d.get('field','other')}] {d.get('image_shows','')[:140]}")
    if context_buffer_up_to_step:
        parts.append("Prior trace context:")
        for step, e in enumerate(context_buffer_up_to_step, 1):
            parts.append(f"  [{step}] {e.get('agent_name','?')} (κ={e.get('confidence','?')})")
            for d in e.get("deltas") or []:
                parts.append(f"      delta[{d.get('field','?')}]: {d.get('image_shows','')[:120]}")
    return "\n".join(parts)


def annotations_from_trace(record: dict) -> List[TraceStepAnnotation]:
    """Expand one trace JSONL record into per-step annotations.

    Returns one TraceStepAnnotation per (step) where step has a defined
    next_agent (i.e. the trace didn't terminate here).
    """
    path = record.get("path") or []
    if not path:
        return []
    buf = record.get("context_buffer") or []
    existing = record.get("existing_kb_at_start") or []
    final_deltas = record.get("final_deltas") or []
    n_final = len(final_deltas)

    out: List[TraceStepAnnotation] = []
    for i, agent_step in enumerate(buf):
        if i + 1 >= len(path):
            # Terminal step — no routing target. Skip for routing supervision.
            continue
        next_agent = path[i + 1]
        if next_agent not in _AGENT_IDX:
            continue
        kappa = (agent_step.get("confidence") or "medium").lower()
        kappa_scalar = _kappa_to_scalar(kappa)
        step_deltas = agent_step.get("deltas") or []
        n_step = len(step_deltas)
        # Epistemic: how much more the trace added after this step.
        epistemic = (
            max(0.0, (n_final - n_step) / max(1, n_final))
            if n_final > 0 else 0.0
        )
        # Aleatoric: 1 - κ scalar (low κ = high irreducible noise).
        aleatoric = 1.0 - kappa_scalar
        # Overconfidence: claimed high but produced nothing.
        oc = bool(kappa == "high" and n_step == 0)
        backtrack = bool(next_agent == "MorphologyAgent"
                         and agent_step.get("agent_name") != "MorphologyAgent")

        ctx_text = _render_context_text(
            crop=record.get("crop", ""),
            disease=record.get("disease", ""),
            state=record.get("state", ""),
            existing_kb=existing,
            context_buffer_up_to_step=buf[:i],
        )

        out.append(TraceStepAnnotation(
            image_path=record.get("image_path", ""),
            crop=record.get("crop", ""),
            disease=record.get("disease", ""),
            state=record.get("state", ""),
            step=i,
            current_agent=agent_step.get("agent_name", "MorphologyAgent"),
            context_text=ctx_text,
            next_agent=next_agent,
            backtrack=backtrack,
            confidence=kappa_scalar,
            epistemic=epistemic,
            aleatoric=aleatoric,
            overconfidence=oc,
            belief_state=agent_step.get("reasoning", "") or "",
            profile_id=record.get("profile_id", ""),
            run_idx=int(record.get("run_idx", 0)),
            n_deltas_at_step=n_step,
            n_deltas_final=n_final,
        ))
    return out


def load_phase0r_traces(path: str | Path) -> List[TraceStepAnnotation]:
    """Read a Phase 0R trace JSONL and expand to per-step annotations."""
    annotations: List[TraceStepAnnotation] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            annotations.extend(annotations_from_trace(rec))
    return annotations


# ---------------------------------------------------------------------------
# Image-aware dataset (lazy load from path)
# ---------------------------------------------------------------------------

class RoutingTraceDataset(Dataset):
    """PyTorch dataset over Phase 0R trace step annotations.

    Lazy-loads images from ``image_path`` on every __getitem__. For
    small caches this is fine; for production, an on-disk image cache
    keyed by hash would be cheaper.
    """

    def __init__(self, annotations: Sequence[TraceStepAnnotation], processor):
        self.annotations = list(annotations)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        ann = self.annotations[idx]
        image = Image.open(ann.image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=ann.context_text,
            return_tensors="pt",
            padding=True,
        )
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor) and v.ndim > 1:
                inputs[k] = v.squeeze(0)
        return {
            "inputs":         inputs,
            "next_agent":     torch.tensor(_AGENT_IDX[ann.next_agent], dtype=torch.long),
            "backtrack":      torch.tensor(float(ann.backtrack), dtype=torch.float32),
            "epistemic":      torch.tensor(ann.epistemic,        dtype=torch.float32),
            "aleatoric":      torch.tensor(ann.aleatoric,        dtype=torch.float32),
            "confidence":     torch.tensor(ann.confidence,       dtype=torch.float32),
            "overconfidence": torch.tensor(float(ann.overconfidence), dtype=torch.float32),
            "belief_state":   ann.belief_state,
        }


# ---------------------------------------------------------------------------
# Image-grouped split (no leakage across image_id)
# ---------------------------------------------------------------------------

def split_annotations(
    annotations: Sequence[TraceStepAnnotation],
    *,
    val_frac: float = 0.1,
    held_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Group by source image_path so all runs / steps of one image stay
    in the same fold. Returns {'train', 'val', 'held'} list-of-annotations.
    """
    import random as _random
    by_image: dict[str, List[TraceStepAnnotation]] = {}
    for ann in annotations:
        by_image.setdefault(ann.image_path, []).append(ann)
    unique = sorted(by_image.keys())
    rng = _random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_val = int(n * val_frac)
    n_held = int(n * held_frac)
    n_train = n - n_val - n_held
    train_ids = set(unique[:n_train])
    val_ids   = set(unique[n_train:n_train + n_val])
    held_ids  = set(unique[n_train + n_val:])

    train_anns, val_anns, held_anns = [], [], []
    for img_id, anns in by_image.items():
        bucket = (train_anns if img_id in train_ids
                  else val_anns if img_id in val_ids
                  else held_anns)
        bucket.extend(anns)
    return {"train": train_anns, "val": val_anns, "held": held_anns,
            "n_images_train": len(train_ids), "n_images_val": len(val_ids),
            "n_images_held": len(held_ids)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OBSERVETrainer:
    """Behavioral-cloning trainer for OBSERVE on Phase 0R traces."""

    def __init__(self, model, lr: float = 1e-4, weight_decay: float = 0.01):
        self.model = model
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn,
        device: torch.device,
    ) -> dict:
        self.model.train()
        sums = {"total": 0.0, "routing": 0.0, "cal": 0.0,
                "cons": 0.0, "oc": 0.0}
        n = 0
        for batch in tqdm(loader, desc="train"):
            self.optimizer.zero_grad()
            out = self.model(
                image=None,                  # processor inputs already in batch['inputs']
                context_text=None,
                return_dict=True,
                _batched_inputs=batch["inputs"],
            ) if hasattr(self.model, "forward") and "_batched_inputs" in \
                self.model.forward.__code__.co_varnames else self.model(
                image=batch["inputs"].get("pixel_values"),
                context_text="",
                return_dict=True,
            )
            losses = loss_fn(
                routing_logits=out["routing_probs"].log() if out["routing_probs"].min() > 0 else out["routing_probs"],
                target_class=batch["next_agent"].to(device),
                epsilon_pred=out["epistemic"],   epsilon_target=batch["epistemic"].to(device),
                aleatoric_pred=out["aleatoric"], aleatoric_target=batch["aleatoric"].to(device),
                confidence_pred=out["confidence"], confidence_target=batch["confidence"].to(device),
                oc_pred=out["oc_prob"],         oc_target=batch["overconfidence"].to(device),
            )
            losses.total.backward()
            self.optimizer.step()
            sums["total"]   += float(losses.total.item())
            sums["routing"] += float(losses.routing.item())
            sums["cal"]     += float(losses.calibration.item())
            sums["cons"]    += float(losses.consistency.item())
            sums["oc"]      += float(losses.overconfidence.item())
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, loss_fn, device: torch.device) -> dict:
        self.model.eval()
        sums = {"total": 0.0, "routing_acc": 0.0}
        n = 0
        for batch in loader:
            out = self.model(
                image=batch["inputs"].get("pixel_values"),
                context_text="",
                return_dict=True,
            )
            pred = out["routing_probs"].argmax(dim=-1).cpu()
            target = batch["next_agent"]
            sums["routing_acc"] += float((pred == target).float().mean().item())
            losses = loss_fn(
                routing_logits=out["routing_probs"].log() if out["routing_probs"].min() > 0 else out["routing_probs"],
                target_class=batch["next_agent"].to(device),
                epsilon_pred=out["epistemic"],   epsilon_target=batch["epistemic"].to(device),
                aleatoric_pred=out["aleatoric"], aleatoric_target=batch["aleatoric"].to(device),
                confidence_pred=out["confidence"], confidence_target=batch["confidence"].to(device),
                oc_pred=out["oc_prob"],         oc_target=batch["overconfidence"].to(device),
            )
            sums["total"] += float(losses.total.item())
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("OBSERVE checkpoint → %s", path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("OBSERVE checkpoint loaded ← %s", path)
