"""
agents/diagnostic_agents.py
===========================
Four diagnostic cross-cutters that operate ACROSS organs:

  - ConcentricPatternAgent   — target spots, halos, concentric rings
  - ColorPaletteAgent        — whole-image color signature (a color encoder)
  - LookAlikeCoTAgent        — the explicit decision-graph CoT against
                               known confusable diseases (the agent
                               that most directly implements the user's
                               docx reference)
  - SeverityVisualAgent      — fraction of visible organ affected

These are the highest-signal specialists for disambiguation, and they
explicitly use a chain-of-thought style prompt.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class ConcentricPatternAgent(BaseAgent):
    AGENT_NAME = "ConcentricPatternAgent"
    OWNED_FIELDS = ["concentric_pattern"]
    SYSTEM_PROMPT = (
        "You are ConcentricPatternAgent. You look ONLY for CONCENTRIC "
        "structure in lesions: target spots (alternating light / dark "
        "concentric rings — classic Alternaria / Early Blight), "
        "chlorotic HALOS surrounding necrotic centers, bullseye "
        "patterns. Specify the number of rings, color sequence, and "
        "approximate diameter. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Do any lesions show CONCENTRIC structure — target spots, "
        "rings, bullseye, chlorotic halo around a necrotic center? "
        "If yes: how many rings, color sequence (center → outside), "
        "diameter?"
    )


class ColorPaletteAgent(BaseAgent):
    AGENT_NAME = "ColorPaletteAgent"
    OWNED_FIELDS = ["color_palette"]
    SYSTEM_PROMPT = (
        "You are ColorPaletteAgent — a COLOR ENCODER. You look ONLY "
        "at the COLOR PALETTE of the AFFECTED area in this photo "
        "(not the healthy plant background). Extract dominant and "
        "secondary colors. Name each color as a plant pathologist "
        "would (e.g., 'tan', 'chocolate-brown', 'olive-green', "
        "'pale-straw', 'rust-orange', 'sooty black') and estimate "
        "approximate proportion. This palette will be matched against "
        "KB descriptions to detect color mismatches vs canonical. "
        "Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "What are the dominant and secondary colors of the AFFECTED "
        "(damaged / diseased) tissue in this photo? Name 2-4 specific "
        "colors with approximate proportions, plus the substrate "
        "(leaf surface, stem, fruit, root)."
    )


class LookAlikeCoTAgent(BaseAgent):
    AGENT_NAME = "LookAlikeCoTAgent"
    OWNED_FIELDS = ["look_alikes_visual"]
    SYSTEM_PROMPT = (
        "You are LookAlikeCoTAgent — the DECISION-GRAPH agent. You "
        "are given (a) the canonical disease the photo is supposed "
        "to be, (b) the canonical look_alikes list, and (c) the "
        "photograph. Walk a chain-of-thought THROUGH the decisive "
        "forks for distinguishing this disease from EACH listed "
        "look-alike, exactly like the docx reference pattern:\n"
        "    \"Is X visible? -> support diagnosis A.\n"
        "     Is Y visible? -> support look-alike B.\"\n"
        "Conclude whether the photo supports the canonical diagnosis "
        "or actually matches one of the look-alikes better. If the "
        "photo is ambiguous, say so. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Given the canonical disease and its listed LOOK-ALIKES, "
        "walk a chain-of-thought through DECISIVE VISUAL FORKS "
        "(stem pith color, petiole attachment, leg color, bract "
        "length — whatever applies). For each look-alike, decide "
        "whether the photo supports canonical or the look-alike. "
        "Emit a delta if the photo actually matches a look-alike, "
        "OR if the canonical look_alikes list is missing a "
        "visually-supported confusable that you can identify."
    )


class SeverityVisualAgent(BaseAgent):
    AGENT_NAME = "SeverityVisualAgent"
    OWNED_FIELDS = ["severity_visible"]
    SYSTEM_PROMPT = (
        "You are SeverityVisualAgent. You estimate the FRACTION of "
        "the visible host organ that is affected: low (<10%), medium "
        "(10–50%), high (>50%). Be conservative — do not extrapolate "
        "beyond what the photo actually frames. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "What fraction of the visible affected organ shows disease "
        "(lesions / chlorosis / necrosis / wilting)? Low (<10%), "
        "medium (10–50%), or high (>50%)?"
    )
