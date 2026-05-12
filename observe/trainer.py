"""
observe/trainer.py
==================
Behavioral-cloning trainer for OBSERVE on Phase 0R trace JSONL.

Per trace step, supervision is derived from the swarm's actual move::

    target_routing       = path[i+1]                       (5-class)
    target_backtrack     = path[i+1]=="MorphologyAgent" AND
                           path[i] != "MorphologyAgent"
    target_confidence    = kappa in {high, medium, low} → {0.9, 0.6, 0.3}
    target_epistemic     = (n_final - n_at_step) / max(1, n_final)
    target_aleatoric     = 1 - kappa_scalar
    target_overconfidence= 1 iff kappa=="high" AND len(deltas)==0

Implementation notes (post-audit fixes)
---------------------------------------
- The previous revision wrapped the backbone in ``torch.no_grad()`` —
  fixed in ``observe/model.py``, gradients now flow through LoRA.
- ``ObserveLoss`` expects RAW logits, not softmaxed probabilities.
  No more ``probs.log()`` hack.
- A custom ``ObserveCollator`` pads token sequences and stacks pixel
  values so ``DataLoader(batch_size>1)`` actually works.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
    image_path:        str
    crop:              str
    disease:           str
    state:             str
    step:              int
    current_agent:     str
    context_text:      str
    # Targets ↓
    next_agent:        str
    backtrack:         bool
    confidence:        float
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
            parts.append(f"  [{step}] {e.get('agent_name','?')} (k={e.get('confidence','?')})")
            for d in e.get("deltas") or []:
                parts.append(f"      delta[{d.get('field','?')}]: {d.get('image_shows','')[:120]}")
    return "\n".join(parts)


def annotations_from_trace(record: dict) -> List[TraceStepAnnotation]:
    """Expand one trace JSONL record into per-step annotations."""
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
            continue
        next_agent = path[i + 1]
        if next_agent not in _AGENT_IDX:
            continue
        kappa = (agent_step.get("confidence") or "medium").lower()
        kappa_scalar = _kappa_to_scalar(kappa)
        step_deltas = agent_step.get("deltas") or []
        n_step = len(step_deltas)
        epistemic = (
            max(0.0, (n_final - n_step) / max(1, n_final))
            if n_final > 0 else 0.0
        )
        aleatoric = 1.0 - kappa_scalar
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
# Dataset — yields raw items; collator does processor + stacking
# ---------------------------------------------------------------------------

class RoutingTraceDataset(Dataset):
    """One sample per agent step. The collator handles processor + batching."""

    def __init__(self, annotations: Sequence[TraceStepAnnotation]):
        self.annotations = list(annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        return {
            "image_path":     ann.image_path,
            "context_text":   ann.context_text,
            "next_agent_idx": _AGENT_IDX[ann.next_agent],
            "backtrack":      float(ann.backtrack),
            "epistemic":      float(ann.epistemic),
            "aleatoric":      float(ann.aleatoric),
            "confidence":     float(ann.confidence),
            "overconfidence": float(ann.overconfidence),
            "belief_state":   ann.belief_state,
        }


# ---------------------------------------------------------------------------
# Collator — turns a list of samples into a model-ready batch
# ---------------------------------------------------------------------------

class ObserveCollator:
    """Build a batch via the processor (images + text) + stack labels.

    Reads images lazily from disk per sample, runs the processor in
    batch mode (which handles tokenizer padding + pixel stacking on
    Qwen2.5-VL), and tacks on label tensors. Returns one big dict the
    model + loss layer consume directly.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples: List[dict]) -> dict:
        from PIL import Image
        images = [Image.open(s["image_path"]).convert("RGB") for s in samples]
        texts  = [s["context_text"] for s in samples]
        inputs = self.processor(
            images=images, text=texts,
            return_tensors="pt", padding=True, truncation=True,
        )

        labels = {
            "next_agent":     torch.tensor([s["next_agent_idx"]    for s in samples], dtype=torch.long),
            "backtrack":      torch.tensor([s["backtrack"]         for s in samples], dtype=torch.float32),
            "epistemic":      torch.tensor([s["epistemic"]         for s in samples], dtype=torch.float32),
            "aleatoric":      torch.tensor([s["aleatoric"]         for s in samples], dtype=torch.float32),
            "confidence":     torch.tensor([s["confidence"]        for s in samples], dtype=torch.float32),
            "overconfidence": torch.tensor([s["overconfidence"]    for s in samples], dtype=torch.float32),
        }
        return {"inputs": dict(inputs), "labels": labels}


# ---------------------------------------------------------------------------
# Image-grouped split (no leakage across image_path)
# ---------------------------------------------------------------------------

def split_annotations(
    annotations: Sequence[TraceStepAnnotation],
    *,
    val_frac: float = 0.1,
    held_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    import random as _random
    by_image: dict[str, List[TraceStepAnnotation]] = {}
    for ann in annotations:
        by_image.setdefault(ann.image_path, []).append(ann)
    unique = sorted(by_image.keys())
    rng = _random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_val  = int(n * val_frac)
    n_held = int(n * held_frac)
    n_train = n - n_val - n_held
    train_ids = set(unique[:n_train])
    val_ids   = set(unique[n_train:n_train + n_val])
    held_ids  = set(unique[n_train + n_val:])

    train_a: List[TraceStepAnnotation] = []
    val_a:   List[TraceStepAnnotation] = []
    held_a:  List[TraceStepAnnotation] = []
    for img_id, anns in by_image.items():
        bucket = (train_a if img_id in train_ids
                  else val_a if img_id in val_ids
                  else held_a)
        bucket.extend(anns)
    return {
        "train": train_a, "val": val_a, "held": held_a,
        "n_images_train": len(train_ids),
        "n_images_val":   len(val_ids),
        "n_images_held":  len(held_ids),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OBSERVETrainer:
    """Behavioral-cloning trainer for OBSERVE on Phase 0R traces.

    Use ``make_loader(annotations, batch_size)`` to get a DataLoader
    that uses the correct collator. ``train_epoch`` and ``validate``
    expect that loader's batch shape.
    """

    def __init__(self, model, lr: float = 1e-4, weight_decay: float = 0.01):
        self.model = model
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )

    def make_loader(
        self,
        annotations: Sequence[TraceStepAnnotation],
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        ds = RoutingTraceDataset(annotations)
        collator = ObserveCollator(self.model.processor)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=collator,
        )

    def _step(self, batch: dict, loss_fn, device: torch.device, train: bool) -> dict:
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in batch["inputs"].items()}
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        out = self.model(inputs)
        losses = loss_fn(
            routing_logits=out["routing_logits"],
            target_class=labels["next_agent"],
            epsilon_logit=out["epistemic_logit"],     epsilon_target=labels["epistemic"],
            aleatoric_logit=out["aleatoric_logit"],   aleatoric_target=labels["aleatoric"],
            confidence_logit=out["confidence_logit"], confidence_target=labels["confidence"],
            oc_logit=out["oc_logit"],                 oc_target=labels["overconfidence"],
        )
        if train:
            losses.total.backward()
            self.optimizer.step()

        pred = out["routing_logits"].argmax(dim=-1)
        acc = float((pred == labels["next_agent"]).float().mean().item())
        return {
            "total":   float(losses.total.item()),
            "routing": float(losses.routing.item()),
            "cal":     float(losses.calibration.item()),
            "cons":    float(losses.consistency.item()),
            "oc":      float(losses.overconfidence.item()),
            "routing_acc": acc,
        }

    def train_epoch(self, loader: DataLoader, loss_fn, device: torch.device) -> dict:
        sums = {"total": 0.0, "routing": 0.0, "cal": 0.0,
                "cons": 0.0, "oc": 0.0, "routing_acc": 0.0}
        n = 0
        for batch in tqdm(loader, desc="train", leave=False):
            stats = self._step(batch, loss_fn, device, train=True)
            for k in sums:
                sums[k] += stats[k]
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader, loss_fn, device: torch.device) -> dict:
        sums = {"total": 0.0, "routing_acc": 0.0}
        n = 0
        for batch in tqdm(loader, desc="val", leave=False):
            stats = self._step(batch, loss_fn, device, train=False)
            sums["total"]       += stats["total"]
            sums["routing_acc"] += stats["routing_acc"]
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
