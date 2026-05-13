"""
agents/severity_agent.py
========================
SeverityAgent — emits deltas for disease advancement at this site and
treatment-relevant stage cues visible in the photograph.

Invoked in parallel with the other specialists; no routing state.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class SeverityAgent(BaseAgent):
    AGENT_NAME = "SeverityAgent"
    OWNED_FIELDS = ["severity", "treatments"]

    SYSTEM_PROMPT = (
        "You are SeverityAgent. Inspect the photograph for disease "
        "advancement (early / moderate / severe / whole-field collapse), "
        "extent at this site, and any treatment-relevant stage cues. "
        "Output strict JSON only — no prose, no markdown."
    )
