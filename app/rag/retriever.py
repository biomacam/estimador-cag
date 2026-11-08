"""Metadata-filtered k-NN retrieval with a relevance threshold (vector only)."""

from __future__ import annotations

import time

import structlog

from app.rag.errors import RetrievalError
from app.rag.schemas import RetrievalResult, RetrievedChunk

log = structlog.get_logger()


def row_to_chunk(row, *, distance: float) -> RetrievedChunk:
    """Map a DB row (vector or lexical) onto the typed retrieval contract."""
    return RetrievedChunk(
        id=row.id,
        content=row.content,
        sector=str(row.metadata_.get("client_sector", "unknown")),
        project_year=int(row.metadata_.get("year", 0)),
        chunk_type=row.chunk_type,
        distance=distance,
        budget_id=row.metadata_.get("budget_id"),
    )


async def search_chunks(
    query_embedding: list[float],
    top_k: int = 10,
    distance_threshold: float = 0.6,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
) -> RetrievalResult:
    """Structural filters narrow the space before the vector ranking; the threshold
    enforces a relevance floor.

    ``low_confidence`` is True (soft-fail) when nothing crosses the threshold, so
    the orchestrator skips generation instead of grounding on irrelevant budgets.

    Raises
    ------
    RetrievalError
        If the vector store cannot be queried.
    """
    from app.dependencies import get_chunk_store
    from app.persistence.database import get_async_session_factory

    session_factory = get_async_session_factory()
    store = get_chunk_store()

    started = time.perf_counter()
    try:
        async with session_factory() as session:
            rows, candidates_evaluated = await store.search_filtered(
                session,
                query_vector=query_embedding,
                top_k=top_k,
                distance_threshold=distance_threshold,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )
    except Exception as exc:  # noqa: BLE001 — DB/connection failure.
        log.error("rag_filtered_search_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        raise RetrievalError("Vector store query failed.") from exc

    chunks = [row_to_chunk(row, distance=float(row.distance)) for row in rows]
    log.info(
        "rag_filtered_search_done",
        results=len(chunks),
        candidates_evaluated=candidates_evaluated,
        top_k=top_k,
        distance_threshold=distance_threshold,
        search_time_ms=int((time.perf_counter() - started) * 1000),
    )
    return RetrievalResult(
        chunks=chunks,
        low_confidence=not chunks,
        candidates_evaluated=candidates_evaluated,
    )
