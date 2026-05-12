"""
observe/
========
OBSERVE: Vision-Language-Action model for epistemic action selection.

Fine-tuned from Qwen2.5-VL-7B with LoRA (r=16, α=32). In the original
paper plan, OBSERVE was trained on Phase 2 routing traces (T1-T5
classification swarm). In the current pipeline (Phase 1-5 retired,
Phase 0R Qwen swarm in place), OBSERVE is retargeted to **delta-mode
supervision**: it learns to imitate the swarm's per-step routing and
delta-emission behavior on Phase 0R trace JSONL.

Heads (model.py — unchanged from paper):
  next_agent (5-class), backtrack b_t, epistemic ε_t, aleatoric α_t,
  calibrated confidence c_t, overconfidence flag OC_t, belief s_t.

Training data:
  Phase 0R writes per-trace records to
  ``$PATHOME_TRACE_DIR/phase0r_traces.jsonl`` (one line per
  (tuple, run)) when the env var is set. ``observe.trainer.load_phase0r_traces``
  expands those into per-step (s, a) annotations.

Inference replaces the N-stochastic-traces swarm with a single OBSERVE
forward pass — 6× faster, calibrated.

NOTE — Decision Transformer (decision_transformer.py) and GRPO (grpo.py)
are restored verbatim from the pre-refactor history but have NOT yet been
ported to the delta-mode reward signal (delta-set F1 + κ ECE). They will
need a reward function adaptation before use. Phase A (BC via
OBSERVETrainer) is the v1 path.
"""

from .model import OBSERVE, EpistemicAction
from .trainer import (
    OBSERVETrainer,
    RoutingTraceDataset,
    TraceStepAnnotation,
    annotations_from_trace,
    load_phase0r_traces,
    split_annotations,
    AGENT_CLASSES,
)
from .inference import OBSERVEInference

__all__ = [
    "OBSERVE", "EpistemicAction",
    "OBSERVETrainer", "RoutingTraceDataset", "TraceStepAnnotation",
    "annotations_from_trace", "load_phase0r_traces", "split_annotations",
    "AGENT_CLASSES",
    "OBSERVEInference",
]
