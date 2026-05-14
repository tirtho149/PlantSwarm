"""
agents/reproductive_agents.py
=============================
Two specialists for reproductive structures: flowers and fruits. Crop
disease diagnosis often hinges on whether a pathogen reaches the
reproductive stage (fruit rots, mummification, flower blights).
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class FlowerAgent(BaseAgent):
    AGENT_NAME = "FlowerAgent"
    OWNED_FIELDS = ["flower"]
    SYSTEM_PROMPT = (
        "You are FlowerAgent. You look ONLY at FLOWERS visible in the "
        "photo: flower blight (browning / wilting of intact flowers), "
        "petal spotting, distortion / aborted flowers, gray mold fuzz "
        "on petals (Botrytis), flower discoloration. If no flowers "
        "are visible return empty deltas with confidence 'low'. Output "
        "strict JSON only."
    )
    FOCUS_QUESTION = (
        "Are flowers visible? If yes: any browning / wilting, petal "
        "spots, distortion, abortion, or fuzzy mold on petals?"
    )


class FruitAgent(BaseAgent):
    AGENT_NAME = "FruitAgent"
    OWNED_FIELDS = ["fruit"]
    SYSTEM_PROMPT = (
        "You are FruitAgent. You look ONLY at FRUITS visible in the "
        "photo: fruit lesions (color, sunken-or-raised, concentric "
        "rings), mummified fruit (dry shriveled), fruit scab (rough "
        "corky patches), soft watery rots, internal browning visible "
        "through cracks. If no fruit is visible return empty deltas "
        "with confidence 'low'. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Are fruits visible? If yes: any lesions (shape, color, "
        "concentric pattern), mummification, scab, soft rot, or "
        "visible internal browning?"
    )
