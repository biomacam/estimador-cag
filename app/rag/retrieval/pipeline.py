"""Recall-then-rerank retrieval pipeline.

``retrieve()`` composes four configurations behind two switches the caller resolves
(request param -> runtime override -> settings):

* ``search_mode="vector"``: dense k-NN only.
* ``search_mode="hybrid"``: dense + lexical branches fused with RRF.
* ``rerank=False``: keep the top ``top_k`` of the recall ordering.
* ``rerank=True``: recall WIDE (``recall_k``), let the cross-encoder rescore and
  keep the top ``rerank_top_n``.

It returns the same :class:`RetrievalResult` as ``search_chunks``.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from app.rag.errors import RetrievalError
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retriever import row_to_chunk
from app.rag.schemas import RetrievalResult, RetrievedChunk

log = structlog.get_logger()

# Distance for a candidate surfaced ONLY by the lexical branch (no cosine distance);
# a display sentinel, ignored by fusion and reranking.
_NO_VECTOR_DISTANCE = 1.0


async def retrieve(
    *,
    query_embedding: list[float],
    query_text: str,
    search_mode: str = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = 60,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker=None,
) -> RetrievalResult:
    """Run hybrid/vector retrieval with optional cross-encoder reranking.

    ``query_text`` feeds the lexical branch and the reranker even when
    ``search_mode`` is ``"vector"``. ``reranker`` is injectable for tests; when
    ``None`` and ``rerank`` is True it comes from the composition root.
    ``low_confidence`` is True (soft-fail) when nothing was retrieved at all.
    """
    from app.dependencies import get_chunk_store
    from app.persistence.database import get_async_session_factory

    session_factory = get_async_session_factory()
    store = get_chunk_store()
    started = time.perf_counter()

    # Recall wide whenever a later stage (fusion or rerank) will re-sort.
    wide = rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else top_k

    try:
        async with session_factory() as session:
            vector_rows, candidates_evaluated = await store.search_filtered(
                session,
                query_vector=query_embedding,
                top_k=vector_limit,
                distance_threshold=distance_threshold,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )
            lexical_rows = []
            if search_mode == "hybrid":
                lexical_rows = await store.search_lexical(
                    session,
                    query_text=query_text,
                    top_k=recall_k,
                    sectors=sectors,
                    project_year_min=project_year_min,
                    project_year_max=project_year_max,
                    chunk_types=chunk_types,
                )
    except Exception as exc:  # noqa: BLE001 — DB/connection failure.
        log.error("rag_hybrid_search_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Vector store query failed.") from exc

    # Vector distance wins over the lexical sentinel when an id is in both branches.
    candidates: dict[int, RetrievedChunk] = {}
    for row in vector_rows:
        candidates[row.id] = row_to_chunk(row, distance=float(row.distance))
    for row in lexical_rows:
        candidates.setdefault(row.id, row_to_chunk(row, distance=_NO_VECTOR_DISTANCE))

    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [[row.id for row in vector_rows], [row.id for row in lexical_rows]],
            k=rrf_k,
        )
        ordered = [candidates[cid] for cid, _score in fused]
    else:
        ordered = [candidates[row.id] for row in vector_rows]

    if rerank and ordered:
        if reranker is None:
            from app.dependencies import get_reranker

            reranker = get_reranker()
        final = await asyncio.to_thread(
            reranker.rerank, query_text, ordered[:recall_k], top_n=rerank_top_n
        )
    else:
        final = ordered[:top_k]

    log.info(
        "rag_retrieve_done",
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_rows),
        lexical_hits=len(lexical_rows),
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        search_time_ms=int((time.perf_counter() - started) * 1000),
    )
    return RetrievalResult(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
    )
