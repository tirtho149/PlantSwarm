"""Qwen visual-symptom swarm — 24 parallel specialists + CoT consolidator.

Each of the 24 specialists owns ONE visual delta field (e.g.
``stem_pith``, ``leaf_chlorosis``, ``concentric_pattern``) and asks
ONE laser-focused question about the photograph. All run in PARALLEL
on the same (image, canonical-KB, existing-KB) input. ``DiagnosisAgent``
(VisualDiagnosisAgent) consolidates the union by walking the look-alike
decision-graph CoT documented in
``Look alike Diseases, weeds and Insect COT with decision graph.docx``.

Non-visual KB fields (pathogen, type_of_disease, treatments, …) are
handled exclusively by Claude in Phase 0; the swarm never emits deltas
for those.
"""

from .base_agent import (
    ALLOWED_DELTA_FIELDS,
    AgentDeltaOutput,
    BaseAgent,
    DELTA_USER_PROMPT,
    parse_agent_output,
)
from .diagnosis_agent import DiagnosisAgent

# 8 leaf specialists
from .leaf_agents import (
    LeafLesionShapeAgent,
    LeafLesionColorAgent,
    LeafLesionTextureAgent,
    LeafChlorosisAgent,
    LeafNecrosisAgent,
    LeafCurlAgent,
    LeafVeinPatternAgent,
    LeafGeometryAgent,
)
# 4 stem specialists
from .stem_agents import (
    StemLesionAgent,
    StemPithAgent,
    StemSurfaceAgent,
    StemDiscolorationAgent,
)
# 2 below-ground specialists
from .root_agents import (
    RootAgent,
    CrownCollarAgent,
)
# 2 reproductive specialists
from .reproductive_agents import (
    FlowerAgent,
    FruitAgent,
)
# 1 pathogen-signs specialist
from .sign_agents import (
    SporulationAgent,
)
# 3 whole-plant pattern specialists
from .pattern_agents import (
    WiltingAgent,
    DefoliationAgent,
    SpatialPatternAgent,
)
# 4 diagnostic cross-cutters (incl. the LookAlikeCoT decision-graph
# agent and a color-encoder)
from .diagnostic_agents import (
    ConcentricPatternAgent,
    ColorPaletteAgent,
    LookAlikeCoTAgent,
    SeverityVisualAgent,
)


# Canonical 24-specialist roster, parallel-invoked by
# ``plantswarm.delta_pipeline.SPECIALIST_CLASSES``.
SPECIALIST_AGENTS = (
    # Leaf (8)
    LeafLesionShapeAgent, LeafLesionColorAgent, LeafLesionTextureAgent,
    LeafChlorosisAgent, LeafNecrosisAgent, LeafCurlAgent,
    LeafVeinPatternAgent, LeafGeometryAgent,
    # Stem (4)
    StemLesionAgent, StemPithAgent, StemSurfaceAgent, StemDiscolorationAgent,
    # Below-ground (2)
    RootAgent, CrownCollarAgent,
    # Reproductive (2)
    FlowerAgent, FruitAgent,
    # Pathogen signs (1)
    SporulationAgent,
    # Patterns (3)
    WiltingAgent, DefoliationAgent, SpatialPatternAgent,
    # Diagnostic cross-cutters (4)
    ConcentricPatternAgent, ColorPaletteAgent,
    LookAlikeCoTAgent, SeverityVisualAgent,
)


__all__ = [
    # base infrastructure
    "BaseAgent",
    "DiagnosisAgent",
    "ALLOWED_DELTA_FIELDS",
    "AgentDeltaOutput",
    "DELTA_USER_PROMPT",
    "parse_agent_output",
    "SPECIALIST_AGENTS",
    # leaf
    "LeafLesionShapeAgent", "LeafLesionColorAgent", "LeafLesionTextureAgent",
    "LeafChlorosisAgent", "LeafNecrosisAgent", "LeafCurlAgent",
    "LeafVeinPatternAgent", "LeafGeometryAgent",
    # stem
    "StemLesionAgent", "StemPithAgent", "StemSurfaceAgent", "StemDiscolorationAgent",
    # below-ground
    "RootAgent", "CrownCollarAgent",
    # reproductive
    "FlowerAgent", "FruitAgent",
    # signs
    "SporulationAgent",
    # patterns
    "WiltingAgent", "DefoliationAgent", "SpatialPatternAgent",
    # diagnostic cross-cutters
    "ConcentricPatternAgent", "ColorPaletteAgent",
    "LookAlikeCoTAgent", "SeverityVisualAgent",
]
