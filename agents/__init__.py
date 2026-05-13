"""Qwen delta-extraction swarm — parallel specialists + consolidator.

The four specialists (Morphology, Symptom, Pathogen, Severity) each
own a slice of canonical KB fields and run in PARALLEL on the same
(image, canonical, existing KB) input. DiagnosisAgent consolidates the
union, deduping overlapping fields and dropping restatements.

No routing, no κ-gated handoff — the swarm is a hypothesis generator;
validation comes from Claude+WebSearch downstream.
"""

from .base_agent import (
    ALLOWED_DELTA_FIELDS,
    AgentDeltaOutput,
    BaseAgent,
    DELTA_USER_PROMPT,
    parse_agent_output,
)
from .diagnosis_agent import DiagnosisAgent
from .morphology_agent import MorphologyAgent
from .pathogen_agent import PathogenAgent
from .severity_agent import SeverityAgent
from .symptom_agent import SymptomAgent

__all__ = [
    "BaseAgent",
    "DiagnosisAgent",
    "MorphologyAgent",
    "SymptomAgent",
    "PathogenAgent",
    "SeverityAgent",
    "ALLOWED_DELTA_FIELDS",
    "AgentDeltaOutput",
    "DELTA_USER_PROMPT",
    "parse_agent_output",
]
