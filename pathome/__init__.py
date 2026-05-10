"""
pathome/
========
PathomeDB: a visual-symptom-centric, geo-aware knowledge base.

Two stores:

  symptoms (SymptomLibrary)   what each (crop, disease) looks like, plus
                              per-state and per-AEZ observation counts
                              (geo prior comes from these counts directly).
  refs (ReferenceLibrary)     held-out reference images for visual retrieval
                              (CLIP + FAISS, climate-weighted).

Predecessor 5-layer modules (layer1_pathway, layer2_manifestation,
layer3_geo, layer4_decision_graph) were retired in favour of the
SymptomProfile aggregation. Only ``layer5_references`` survives — its
ReferenceLibrary continues to back ``db.refs``.
"""

from .database import GeoPriorResult, PathomeDB
from .layer5_references import ReferenceLibrary
from .symptoms import (
    CanonicalDisease,
    Citation,
    RegionalDelta,
    RegionalObservation,
    SwarmObservations,
    SymptomLibrary,
    SymptomProfile,
)

__all__ = [
    "PathomeDB",
    "GeoPriorResult",
    "SymptomLibrary",
    "SymptomProfile",
    "CanonicalDisease",
    "RegionalObservation",
    "RegionalDelta",
    "Citation",
    "SwarmObservations",
    "ReferenceLibrary",
]
