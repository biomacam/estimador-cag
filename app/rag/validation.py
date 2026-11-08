"""Post-generation checks for grounded estimates.

* ``validate_citations``: every cited ``source_id`` must be a chunk that was
  actually retrieved; fabricated ids trigger one corrective retry upstream.
* ``check_coherence``: the ``insufficient`` confidence level has a strict shape
  (no numbers, an explanation present).
"""

from __future__ import annotations

from app.rag.schemas import Estimate, RetrievedChunk


def validate_citations(estimate: Estimate, retrieved_chunks: list[RetrievedChunk]) -> list[int]:
    """Return the sorted, de-duplicated cited ids that were never retrieved.

    Checks both the top-level ``sources`` and the per-task ``modules[].tasks[].sources``.
    """
    valid_ids = {chunk.id for chunk in retrieved_chunks}

    cited_ids: set[int] = {citation.source_id for citation in estimate.sources}
    for module in estimate.modules:
        for task in module.tasks:
            cited_ids.update(task.sources)

    return sorted(cited_ids - valid_ids)


def check_coherence(estimate: Estimate) -> bool:
    """Whether the confidence level matches the content of the estimate.

    ``insufficient`` requires null totals, no modules and a non-empty explanation;
    every other level is always coherent here.
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
