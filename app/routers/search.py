"""HTTP endpoint for searching persisted embedding chunks."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse
from app.embedding_pipeline.search import search_chunks
from app.persistence.database import get_async_session

log = structlog.get_logger()
router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={500: {"description": "Embedding or vector search failed."}},
)
async def search(
    request: SearchRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    session: AsyncSession = Depends(get_async_session),
) -> SearchResponse:
    """Return the top-k persisted chunks nearest to the query embedding."""
    if embedder is None:
        log.error("vector_search_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        response = await search_chunks(request, session, embedder)
    except Exception as exc:
        log.error(
            "vector_search_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to search chunks.") from exc

    log.info(
        "vector_search_done",
        query_length=len(request.query),
        k=request.k,
        results=len(response.results),
        search_time_ms=response.search_time_ms,
    )
    return response