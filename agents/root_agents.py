"""
agents/root_agents.py
=====================
Two specialists for the below-ground / soil-line region: roots (when
exposed in the photo) and the crown / collar transition zone. Both are
high-value targets in the look-alike CoT — root cysts and blue fungal
masses are decisive forks for SCN vs IDC and SDS vs BSR.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class RootAgent(BaseAgent):
    AGENT_NAME = "RootAgent"
    OWNED_FIELDS = ["root_visible"]
    SYSTEM_PROMPT = (
        "You are RootAgent. You look ONLY at roots VISIBLE in the "
        "photo. Diagnostic targets: root rot (color, extent), small "
        "lemon-shaped CYSTS on roots (mid-late SCN), BLUE fungal "
        "masses or blue spore masses near the crown / taproot (SDS), "
        "root hypertrophy / clubbing, presence of nodules. If no "
        "roots are visible, return empty deltas with confidence 'low'. "
        "Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Are roots visible? If yes: any rot (color, partial vs total), "
        "small lemon-shaped CYSTS on root surface (yellow / white), "
        "BLUE fungal / spore masses near crown / taproot, hypertrophy "
        "or clubbing, or visible nodules?"
    )


class CrownCollarAgent(BaseAgent):
    AGENT_NAME = "CrownCollarAgent"
    OWNED_FIELDS = ["crown_collar"]
    SYSTEM_PROMPT = (
        "You are CrownCollarAgent. You look ONLY at the CROWN / COLLAR "
        "region — the zone where stem meets soil. Diagnostic targets: "
        "crown rot, girdling at soil line, dark sunken cankers at the "
        "collar, white mycelium / sclerotia (mustard-seed-like) at "
        "the soil line (Southern Blight / Sclerotinia). Output strict "
        "JSON only."
    )
    FOCUS_QUESTION = (
        "Look at the crown / collar (where stem meets soil). Any rot, "
        "girdling, sunken collar canker, white mycelium, or "
        "mustard-seed-shaped sclerotia at the soil line?"
    )
