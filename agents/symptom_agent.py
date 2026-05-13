"""
agents/symptom_agent.py
=======================
SymptomAgent — emits deltas for spread pattern across the canopy and
any additional diagnostic features the photograph reveals.

Invoked in parallel with the other specialists; no routing state.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class SymptomAgent(BaseAgent):
    AGENT_NAME = "SymptomAgent"
    OWNED_FIELDS = ["spread_pattern", "diagnostic_features"]

    SYSTEM_PROMPT = (
        "You are SymptomAgent. Inspect the photograph for canopy "
        "distribution patterns (lower / upper / scattered / uniform), "
        "spread direction, and any additional diagnostic features beyond "
        "what canonical describes. Output strict JSON only — no prose, "
        "no markdown."
    )
