"""
pathome/
========
Schema definitions for the PathomeDB seed JSON.

After the Phase 2-5 retirement, ``pathome`` is the **schema documentation
layer** for the seed JSON produced by Phase 0 (canonical KB via Claude)
and Phase 0R (regional deltas via the Qwen swarm). Downstream consumers
import the dataclasses to load and inspect ``symptoms_seed.json``.

The full pipeline is now:

  Phase 0   pathome_kb (claude)         canonical KB
  Phase 0R  plantswarm/delta_pipeline   regional deltas (qwen swarm)
            ↓
            symptoms_seed.json   ← terminal deliverable
"""

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
    "SymptomLibrary",
    "SymptomProfile",
    "CanonicalDisease",
    "RegionalObservation",
    "RegionalDelta",
    "Citation",
    "SwarmObservations",
]
