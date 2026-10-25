"""HTTP boundary for structural chunking and embedding ingestion."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.ingestion import DuplicateDocumentError, ingest_document
from app.embedding_pipeline.schemas import (
    IngestConflictResponse,
    IngestRequest,
    IngestResponse,
)
from app.persistence.database import get_async_session

log = structlog.get_logger()
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        409: {"model": IngestConflictResponse, "description": "Document already ingested."},
        500: {"description": "Embedding service unavailable or ingestion failed."},
    },
)
async def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
    session: AsyncSession = Depends(get_async_session),
) -> IngestResponse | JSONResponse:
    """Persist one document and its embedded chunks atomically."""
    if embedder is None:
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        response = await ingest_document(request, session, chunker, embedder)
    except DuplicateDocumentError as exc:
        return JSONResponse(
            status_code=409,
            content=IngestConflictResponse(
                detail="Document already ingested", document_id=exc.document_id
            ).model_dump(),
        )
    except Exception as exc:
        log.error(
            "embeddings_ingest_failed",
            reason="ingestion_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to ingest document.") from exc

    log.info("embeddings_ingest_done", **response.model_dump())
    return response


@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={
        400: {"description": "Unknown chunking strategy."},
        500: {"description": "Missing API key, embedding or chunking failure."},
    },
)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets: stats plus top-k cosine search."""
    if embedder is None:
        log.error("embeddings_compare_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc.args[0]}") from exc
    except RuntimeError as exc:
        log.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(request.budgets, request.queries, request.top_k)
    except Exception as exc:
        log.error(
            "embeddings_compare_failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run chunking comparison.") from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)