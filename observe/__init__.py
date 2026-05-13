"""
observe/
========
OBSERVE: Qwen2.5-VL + LoRA distilled student trained on Phase 0R
per-pass uncertainty signals.

Algorithm-1 routing was removed: there is no routing decision for the
student to learn. OBSERVE now predicts only uncertainty / confidence
over a single-pass delta extraction (4 heads: epistemic, aleatoric,
confidence, overconfidence).
"""

from .model import OBSERVE, EpistemicState
from .trainer import (
    OBSERVETrainer,
    PassAnnotation,
    PassDataset,
    annotation_from_pass,
    load_phase0r_traces,
    split_annotations,
)

__all__ = [
    "OBSERVE", "EpistemicState",
    "OBSERVETrainer", "PassAnnotation", "PassDataset",
    "annotation_from_pass", "load_phase0r_traces", "split_annotations",
]
