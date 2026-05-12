"""PlantSwarm: Qwen-driven regional delta extraction for PathomeDB.

The Qwen swarm reads canonical KB text plus a single Bugwood field
photograph and emits state-specific deltas — additions or contradictions
backed by image evidence. See ``plantswarm.delta_pipeline``.
"""

from .delta_pipeline import (
    build_client_from_env,
    flatten_canonical,
    run_batch,
    run_for_state,
)

__all__ = [
    "build_client_from_env",
    "flatten_canonical",
    "run_batch",
    "run_for_state",
]
