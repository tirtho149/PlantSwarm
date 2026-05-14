"""
agents/sign_agents.py
=====================
Signs of the pathogen itself (not the host's response). Pathogen
SIGNS are usually higher-information than host SYMPTOMS for ID — e.g.
seeing fungal sporulation eliminates bacterial and viral hypotheses.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class SporulationAgent(BaseAgent):
    AGENT_NAME = "SporulationAgent"
    OWNED_FIELDS = ["sporulation"]
    SYSTEM_PROMPT = (
        "You are SporulationAgent. You look ONLY at SIGNS of the "
        "pathogen itself — material that is NOT the plant's response "
        "but the pathogen's own structures: visible mycelium (white / "
        "gray fluffy), spore masses (powdery / dusty), fruiting bodies "
        "(pycnidia as black dots, perithecia, apothecia), sclerotia "
        "(mustard-seed-shaped resting structures), bacterial OOZE "
        "(viscous droplets — often amber or milky on stem / leaf), "
        "rust pustules (raised orange / brown / yellow / black spots). "
        "Specify color, density, and substrate (which leaf / stem / "
        "soil surface). Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Any visible PATHOGEN SIGNS (not host symptoms)? Mycelium, "
        "spore masses, pycnidia / fruiting bodies, sclerotia, "
        "bacterial ooze, rust pustules? If yes: color, density, "
        "where on the plant."
    )
