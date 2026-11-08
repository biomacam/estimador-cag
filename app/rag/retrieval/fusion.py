"""Reciprocal Rank Fusion (RRF): combine independent rankings by position, not score.

The vector branch (cosine distance, lower = better) and the lexical branch
(``ts_rank_cd``, higher = better) produce incomparable scores, so only the order
each branch puts documents in is trusted. A document scores ``sum(1 / (k + rank))``
over every ranking it appears in (rank is 0-based); 60 is the Cormack et al. default.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several best-first id rankings into ``(chunk_id, fused_score)``, best first.

    Duplicates within one ranking count only at their best position. Ties break by
    ascending id so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("RRF smoothing constant k must be positive")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for position, chunk_id in enumerate(ranking):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
