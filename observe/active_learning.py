"""
observe/active_learning.py
==========================
Cross-crop active learning loop (paper §7.4 + Table active).

OBSERVE's epsilon_t (resolvable / epistemic uncertainty) drives expert
queries; alpha_t (irreducible / aleatoric uncertainty) does NOT.
Querying high-alpha cases wastes expert effort because labels do not improve
routing — they confirm a genuinely ambiguous image.

Convergence target (paper Table active):
    iter 0  zero-shot:    35-50% T3 F1, 60-65% query rate
    iter 3  converged:    80-87% T3 F1, <8%   query rate, ~950 total labels
    supervised baseline:  5-10k labels for comparable performance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class ALCandidate:
    """One unlabeled image awaiting a possible expert query."""
    image_id: str
    epsilon: float                   # OBSERVE-predicted resolvable uncertainty
    aleatoric: float                 # for filtering, NOT scoring
    confidence: float
    extra: Dict = None               # any caller-provided metadata


@dataclass
class ALDecision:
    image_id: str
    query_expert: bool
    rank: int                        # 0 = highest priority
    score: float


def select_queries(
    candidates: Sequence[ALCandidate],
    label_budget: int,
    *,
    epsilon_min: float = 0.4,
    aleatoric_max: float = 0.7,
) -> List[ALDecision]:
    """
    Rank candidates by epsilon, drop those whose aleatoric exceeds
    ``aleatoric_max`` (irreducible — querying wastes labels), and select the
    top ``label_budget``.

    Returns a list of ALDecision in the original candidate order; entries
    with ``query_expert=False`` were either skipped (alpha too high) or
    fell below the budget.
    """
    decisions: Dict[str, ALDecision] = {}

    eligible = [
        c for c in candidates
        if c.epsilon >= epsilon_min and c.aleatoric <= aleatoric_max
    ]
    eligible.sort(key=lambda c: c.epsilon, reverse=True)

    chosen_ids = {c.image_id for c in eligible[:label_budget]}

    for rank_idx, c in enumerate(eligible):
        decisions[c.image_id] = ALDecision(
            image_id=c.image_id,
            query_expert=c.image_id in chosen_ids,
            rank=rank_idx,
            score=float(c.epsilon),
        )

    # Anything not eligible: emit a non-query decision with rank=-1
    for c in candidates:
        if c.image_id not in decisions:
            decisions[c.image_id] = ALDecision(
                image_id=c.image_id, query_expert=False,
                rank=-1, score=float(c.epsilon),
            )

    return [decisions[c.image_id] for c in candidates]


def expected_query_rate(
    candidates: Iterable[ALCandidate],
    *,
    epsilon_min: float = 0.4,
    aleatoric_max: float = 0.7,
) -> float:
    cs = list(candidates)
    if not cs:
        return 0.0
    eligible = sum(
        1 for c in cs
        if c.epsilon >= epsilon_min and c.aleatoric <= aleatoric_max
    )
    return eligible / len(cs)


# ---------------------------------------------------------------------------
# Top-level loop
# ---------------------------------------------------------------------------

def run_active_learning(
    score_pool: Callable[[], List[ALCandidate]],
    relabel_and_train: Callable[[List[str]], None],
    *,
    iterations: int = 3,
    label_budget_per_iter: int = 350,
    epsilon_min: float = 0.4,
    aleatoric_max: float = 0.7,
) -> List[List[ALDecision]]:
    """
    Run the convergence loop. ``score_pool`` returns current OBSERVE
    epsilon/alpha/confidence for every unlabeled image; ``relabel_and_train``
    accepts the chosen image_ids, fetches expert labels, and re-trains
    OBSERVE one round (Phase B over augmented traces).

    Returns the per-iteration list of decisions for inspection.
    """
    log: List[List[ALDecision]] = []
    for _ in range(iterations):
        candidates = score_pool()
        decisions = select_queries(
            candidates,
            label_budget=label_budget_per_iter,
            epsilon_min=epsilon_min,
            aleatoric_max=aleatoric_max,
        )
        chosen = [d.image_id for d in decisions if d.query_expert]
        relabel_and_train(chosen)
        log.append(decisions)
    return log
