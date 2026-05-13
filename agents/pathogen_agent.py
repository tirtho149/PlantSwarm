"""
agents/pathogen_agent.py
========================
PathogenAgent — emits deltas for look-alikes the image could support
and type-of-disease nuance the image clarifies.

Invoked in parallel with the other specialists; no routing state.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class PathogenAgent(BaseAgent):
    AGENT_NAME = "PathogenAgent"
    OWNED_FIELDS = ["look_alikes", "type_of_disease"]

    SYSTEM_PROMPT = (
        "You are PathogenAgent. Inspect the photograph for signs that "
        "could be confused with another disease (look-alikes) and for "
        "type-of-disease nuance the image clarifies. Output strict JSON "
        "only — no prose, no markdown."
    )
