"""
agents/morphology_agent.py
==========================
MorphologyAgent — emits deltas for lesion morphology, affected plant
organs, and diagnostic features the photograph shows but canonical
text doesn't capture (or contradicts).

Invoked in parallel with the other specialists; no routing state.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class MorphologyAgent(BaseAgent):
    AGENT_NAME = "MorphologyAgent"
    OWNED_FIELDS = ["lesion_morphology", "affected_organs", "diagnostic_features"]

    SYSTEM_PROMPT = (
        "You are MorphologyAgent. Inspect the photograph for lesion shape, "
        "size, margin, colour, surface texture; which plant organs are "
        "affected; and any diagnostic features visible in this specific "
        "image. Output strict JSON only — no prose, no markdown."
    )
