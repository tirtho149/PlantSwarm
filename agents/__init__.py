"""Qwen delta-extraction swarm.

Five agents over Qwen2.5-VL-7B-Instruct. Four specialists each own a
slice of canonical KB fields and emit candidate deltas for those fields
based on a single Bugwood photograph; DiagnosisAgent consolidates the
union, dedupes overlapping fields, and drops restatements of canonical.
"""

from .base_agent import (
    ALLOWED_DELTA_FIELDS,
    BaseAgent,
    DELTA_USER_PROMPT,
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
    "DELTA_USER_PROMPT",
]
