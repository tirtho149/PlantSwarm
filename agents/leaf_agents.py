"""
agents/leaf_agents.py
=====================
Eight leaf-feature specialists. Each owns ONE delta field and asks ONE
focused visual question, following the CoT pattern from
``Look alike Diseases, weeds and Insect COT with decision graph.docx``
(e.g., "Look at several mid-canopy leaves. Are they diamond-shaped with
petioles ≥ leaf length? Then Palmer amaranth.").

These eight cover the leaf surface, which is where the largest visual-
signature variance lives across plant diseases.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Lesion sub-features (shape / color / texture) — three orthogonal axes
# ---------------------------------------------------------------------------

class LeafLesionShapeAgent(BaseAgent):
    AGENT_NAME = "LeafLesionShapeAgent"
    OWNED_FIELDS = ["leaf_lesion_shape"]
    SYSTEM_PROMPT = (
        "You are LeafLesionShapeAgent. You look ONLY at the GEOMETRY "
        "of lesions on leaves: are they circular, angular (vein-limited), "
        "irregular, elongated along the leaf, ring-shaped, or v-shaped "
        "at the leaf margin? Ignore color and texture — those are owned "
        "by sibling agents. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Look at any visible leaf lesions. What is their dominant SHAPE "
        "(circular / angular / irregular / elongated / v-margin / ring)? "
        "Is the shape constrained by leaf veins or does it cross them?"
    )


class LeafLesionColorAgent(BaseAgent):
    AGENT_NAME = "LeafLesionColorAgent"
    OWNED_FIELDS = ["leaf_lesion_color"]
    SYSTEM_PROMPT = (
        "You are LeafLesionColorAgent. You look ONLY at the COLOR of "
        "lesions on leaves: primary lesion-area color, halo color if "
        "any, and whether lesion centers differ from margins. Use "
        "specific descriptors (e.g., 'tan center with chocolate-brown "
        "margin and chlorotic yellow halo') — not generic words like "
        "'dark'. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Describe the lesion color palette: lesion-CENTER color, "
        "lesion-MARGIN color, and HALO color if a halo is visible. Be "
        "specific — name the colors as a plant pathologist would."
    )


class LeafLesionTextureAgent(BaseAgent):
    AGENT_NAME = "LeafLesionTextureAgent"
    OWNED_FIELDS = ["leaf_lesion_texture"]
    SYSTEM_PROMPT = (
        "You are LeafLesionTextureAgent. You look ONLY at the SURFACE "
        "TEXTURE of leaf lesions: sunken vs raised, smooth vs fuzzy / "
        "felty / waxy, dry vs water-soaked, papery vs leathery, "
        "shot-hole (necrotic centers dropping out). Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "What is the lesion SURFACE TEXTURE? Sunken or raised? Smooth, "
        "fuzzy, felty, waxy, or water-soaked? Any shot-hole drop-out "
        "of necrotic centers?"
    )


# ---------------------------------------------------------------------------
# Whole-leaf pattern sub-features
# ---------------------------------------------------------------------------

class LeafChlorosisAgent(BaseAgent):
    AGENT_NAME = "LeafChlorosisAgent"
    OWNED_FIELDS = ["leaf_chlorosis"]
    SYSTEM_PROMPT = (
        "You are LeafChlorosisAgent. You look ONLY at YELLOWING "
        "patterns on leaves: interveinal (yellow between veins with "
        "veins staying green — classic for IDC / Mn-deficiency / some "
        "viruses), marginal, generalized, mosaic / mottling (viral), "
        "or vein-banding. Identify which TYPE of chlorosis is present "
        "and where it is in the canopy (youngest leaves? oldest? "
        "scattered?). Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Is there chlorosis? If yes, what TYPE: interveinal, marginal, "
        "generalized, mosaic / mottling, vein-banding, or none? Where "
        "in the canopy is it most pronounced — youngest leaves, oldest "
        "leaves, or scattered?"
    )


class LeafNecrosisAgent(BaseAgent):
    AGENT_NAME = "LeafNecrosisAgent"
    OWNED_FIELDS = ["leaf_necrosis"]
    SYSTEM_PROMPT = (
        "You are LeafNecrosisAgent. You look ONLY at DEAD-TISSUE "
        "patterns on leaves (brown / black / bleached): is necrosis at "
        "the leaf TIP, leaf MARGIN, in DISCRETE SPOTS, EXTENDING "
        "INWARD from a vein, or covering the WHOLE LEAF? Note whether "
        "necrotic tissue retains shape or has dropped out (shot-hole). "
        "Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Is there leaf necrosis (dead brown / black / bleached tissue)? "
        "If yes, what's the SPATIAL DISTRIBUTION on the leaf: tip, "
        "margin, discrete spots, vein-extending, whole leaf?"
    )


class LeafCurlAgent(BaseAgent):
    AGENT_NAME = "LeafCurlAgent"
    OWNED_FIELDS = ["leaf_curl"]
    SYSTEM_PROMPT = (
        "You are LeafCurlAgent. You look ONLY at leaf SHAPE DISTORTION: "
        "curling (up / down / inward), cupping, puckering, blistering, "
        "thickened-and-leathery look. These are diagnostic for many "
        "viruses, mites, and a few fungal diseases (e.g., peach leaf "
        "curl). Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Are leaves curled, cupped, puckered, blistered, or thickened? "
        "If yes, describe the direction (up / down / inward) and which "
        "part of the leaf is most affected."
    )


class LeafVeinPatternAgent(BaseAgent):
    AGENT_NAME = "LeafVeinPatternAgent"
    OWNED_FIELDS = ["leaf_vein_pattern"]
    SYSTEM_PROMPT = (
        "You are LeafVeinPatternAgent. You look ONLY at the VEINS "
        "themselves: vein clearing (veins translucent / paler than "
        "blade), vein necrosis (veins darker / dead), and whether "
        "lesions or chlorosis CROSS veins or stop at them. Veins are "
        "the most diagnostic feature for viruses and several leaf "
        "spots. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Look closely at the LEAF VEINS. Are they clearing (translucent), "
        "necrotic (dark / dead), or normal? Do lesions or chlorosis "
        "stop at veins (vein-limited) or cross them?"
    )


class LeafGeometryAgent(BaseAgent):
    AGENT_NAME = "LeafGeometryAgent"
    OWNED_FIELDS = ["leaf_geometry"]
    SYSTEM_PROMPT = (
        "You are LeafGeometryAgent. You look ONLY at the LEAF SHAPE "
        "for look-alike disambiguation: length-to-width ratio, "
        "diamond / lanceolate / spade / ovate, serration / lobation, "
        "petiole length vs blade length, tip notch / hair. This is "
        "the decisive vegetative fork for Palmer amaranth vs waterhemp "
        "and similar weed look-alikes. Output strict JSON only."
    )
    FOCUS_QUESTION = (
        "Describe the overall LEAF SHAPE: diamond / lanceolate / spade / "
        "ovate? Length-to-width ratio? Petiole length vs blade length? "
        "Any tip notch, single tip hair, or serration?"
    )
