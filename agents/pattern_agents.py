"""
agents/pattern_agents.py
========================
Three whole-plant / whole-field pattern specialists. These see beyond
single organs and observe TOPOLOGY of damage — which is often more
diagnostic than the individual lesion morphology.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class WiltingAgent(BaseAgent):
    AGENT_NAME = "WiltingAgent"
    OWNED_FIELDS = ["wilting"]
    SYSTEM_PROMPT = (
        "You are WiltingAgent. You look ONLY at WILTING TOPOLOGY: is "
        "the whole plant wilting uniformly (root rot / vascular), or "
        "only ONE branch wilting (Verticillium-style hemispheric), "
        "or wilting on ONE SIDE of the plant only? Time-of-day "
        "wilting (turgor loss at midday) is less informative — note "
        "if it looks structural vs recoverable. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Is the plant wilting? If yes: TOPOLOGY — whole plant, one "
        "side, one branch, hemispheric (half-leaves)? Does it look "
        "structural (collapsed) or just midday turgor loss?"
    )


class DefoliationAgent(BaseAgent):
    AGENT_NAME = "DefoliationAgent"
    OWNED_FIELDS = ["defoliation"]
    SYSTEM_PROMPT = (
        "You are DefoliationAgent. You look ONLY at LEAF-DROP patterns "
        "and what STAYS on the stem. Critical fork in the SDS vs BSR "
        "CoT: do PETIOLES REMAIN ATTACHED after the blade has dropped "
        "(strong SDS evidence) or do leaves AND petioles drop together "
        "(BSR or other)? Also note: which canopy age class is "
        "defoliating (oldest, youngest, scattered)? Output strict "
        "JSON only."
    )
    FOCUS_QUESTION = (
        "Any visible defoliation? Do PETIOLES remain attached to the "
        "stem after blades have dropped (bare-petiole skeletons)? "
        "Which canopy layer is most defoliated (oldest / youngest / "
        "scattered)?"
    )


class SpatialPatternAgent(BaseAgent):
    AGENT_NAME = "SpatialPatternAgent"
    OWNED_FIELDS = ["spatial_pattern"]
    SYSTEM_PROMPT = (
        "You are SpatialPatternAgent. You look ONLY at DISTRIBUTION "
        "of damage in the visible scene — within a single canopy "
        "(top-down, bottom-up, scattered), or across multiple plants "
        "if the photo includes them (patches along tillage lines, "
        "circles expanding, edge-of-field, low spots). Output strict "
        "JSON only."
    )
    FOCUS_QUESTION = (
        "How is damage DISTRIBUTED in the visible scene? Within one "
        "canopy: top-down, bottom-up, scattered? Across plants: "
        "patches, circles, tillage-line-aligned, edge-of-field, "
        "low-spot-correlated?"
    )
