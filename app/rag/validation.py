"""Post-generation checks for grounded estimates.

Two independent guards run after the LLM returns an :class:`Estimate`:

* :func:`verify_citations` — line-level, programmatic citation verification.
  Every cited ``chunk_id`` (per-line :class:`SourceReference` and the
  estimate-global :class:`SourceCitation`) must resolve to a chunk that was
  actually retrieved. Ids that were never in the context are *dangling*. The
  returned :class:`CitationReport` separates correctly grounded lines, dangling
  citations and lines explicitly marked as having no sufficient source data.
* :func:`check_coherence` — the ``insufficient`` confidence level has a strict
  shape (no numbers, an explanation present).
"""

from __future__ import annotations

from app.rag.schemas import (
    CitationReport,
    Estimate,
    LineCitation,
    RetrievedChunk,
)


def verify_citations(
    estimate: Estimate,
    retrieved_chunk_ids: set[str],
) -> CitationReport:
    """Verify every citation against the chunks actually handed to the LLM.

    Walks each estimate line (``modules[].tasks[]``) and checks that every
    ``chunk_id`` it cites is present in ``retrieved_chunk_ids``. The
    estimate-global ``sources`` are checked too, so a fabricated id cannot hide
    at either level.

    ``dangling_citations`` in the result is the sorted, de-duplicated set of
    cited ids that were never retrieved; an empty list means every citation is
    real.
    """
    lines: list[LineCitation] = []
    verified = 0
    dangling: set[str] = set()
    grounded_lines = dangling_lines = insufficient_lines = 0

    for module in estimate.modules:
        for task in module.tasks:
            cited = [ref.chunk_id for ref in task.sources]
            line_dangling = [cid for cid in cited if cid not in retrieved_chunk_ids]
            verified += len(cited) - len(line_dangling)
            dangling.update(line_dangling)

            if not task.grounded:
                status = "insufficient"
                insufficient_lines += 1
            elif line_dangling:
                status = "dangling"
                dangling_lines += 1
            else:
                status = "grounded"
                grounded_lines += 1

            lines.append(
                LineCitation(
                    module=module.name,
                    component=task.name,
                    status=status,
                    cited_chunk_ids=cited,
                    dangling_chunk_ids=line_dangling,
                )
            )

    for citation in estimate.sources:
        cid = str(citation.source_id)
        if cid in retrieved_chunk_ids:
            verified += 1
        else:
            dangling.add(cid)

    return CitationReport(
        total_lines=len(lines),
        grounded_lines=grounded_lines,
        dangling_lines=dangling_lines,
        insufficient_lines=insufficient_lines,
        verified_citations=verified,
        dangling_citations=sorted(dangling),
        lines=lines,
    )


def verify_citations_for_chunks(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationReport:
    """Convenience wrapper: derive the id set from the retrieved chunks."""
    return verify_citations(estimate, {str(chunk.id) for chunk in retrieved_chunks})


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
