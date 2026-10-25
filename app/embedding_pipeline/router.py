"""HTTP boundary for structural chunking and embedding ingestion."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={500: {"description": "Embedding service unavailable or embedding API failure."}},
)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> IngestResponse:
    """Chunk budgets and return embedding vectors plus aggregate stats."""
    if embedder is None:
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embeddings_ingest_received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )
    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd(total_tokens),
    )
    log.info("embeddings_ingest_done", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)


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