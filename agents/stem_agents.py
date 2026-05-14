"""
agents/stem_agents.py
=====================
Four stem-feature specialists. Stems carry the most DECISIVE forks in
the look-alike CoT — e.g., "split the lower stem: is the pith white
(SDS) or brown / cardboard-like (BSR)?".

These specialists assume the photo COULD show a cut stem, but they
must also handle photos where only the intact outer stem is visible.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class StemLesionAgent(BaseAgent):
    AGENT_NAME = "StemLesionAgent"
    OWNED_FIELDS = ["stem_lesion"]
    SYSTEM_PROMPT = (
        "You are StemLesionAgent. You look ONLY at lesions on the "
        "OUTER stem surface: cankers (sunken-and-rough), girdling "
        "(lesion encircling stem and killing tissue above it), "
        "sunken vs raised, dark vs bleached, dry vs water-soaked. "
        "Note location (basal, mid-stem, terminal) and whether the "
        "lesion is on a main stem or a branch. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Are there lesions / cankers / girdling on the OUTER stem? "
        "If yes: where on the stem (basal, mid, terminal), shape "
        "(sunken / raised / encircling), and color?"
    )


class StemPithAgent(BaseAgent):
    AGENT_NAME = "StemPithAgent"
    OWNED_FIELDS = ["stem_pith"]
    SYSTEM_PROMPT = (
        "You are StemPithAgent. You look ONLY at the INTERNAL pith of "
        "a SPLIT stem. This is the DECISIVE fork in the SDS-vs-BSR CoT: "
        "white pith with brown outer vascular tissue => SDS; brown / "
        "cardboard-like continuous pith through internodes => BSR. If "
        "the photo does NOT show a split stem, say so explicitly and "
        "return an empty deltas list with confidence 'low'. Output "
        "strict JSON only."
    )
    FOCUS_QUESTION = (
        "Is a cut / split stem visible in this photo? If yes: what "
        "color is the PITH (white, light, brown, cardboard-brown, "
        "continuous through internodes)? What color is the outer "
        "vascular tissue / cortex?"
    )


class StemSurfaceAgent(BaseAgent):
    AGENT_NAME = "StemSurfaceAgent"
    OWNED_FIELDS = ["stem_surface"]
    SYSTEM_PROMPT = (
        "You are StemSurfaceAgent. You look ONLY at surface FEATURES "
        "of the stem that are NOT cankers: galls (tumors), blisters, "
        "scabs, exudates / ooze, fruiting bodies emerging through the "
        "epidermis. These are signs more than symptoms. Output strict "
        "JSON only."
    )
    FOCUS_QUESTION = (
        "On the stem surface — apart from any lesions — are there "
        "galls, blisters, scabs, ooze / exudates, or fruiting bodies "
        "breaking through the bark / epidermis? Describe color, "
        "texture, and density."
    )


class StemDiscolorationAgent(BaseAgent):
    AGENT_NAME = "StemDiscolorationAgent"
    OWNED_FIELDS = ["stem_discoloration"]
    SYSTEM_PROMPT = (
        "You are StemDiscolorationAgent. You look ONLY at COLOR "
        "PATTERNS on or through the stem surface: vascular streaks "
        "visible through translucent bark, blackleg-style black "
        "blotches, water-soaked dark streaking extending from a "
        "lesion. Do NOT report lesions themselves (StemLesionAgent "
        "owns that). Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Apart from any discrete lesions, are there COLOR PATTERNS "
        "on or through the stem — vascular streaking, dark blotches, "
        "water-soaked streaks extending from a lesion edge?"
    )
