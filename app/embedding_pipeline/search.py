"""Cosine-similarity search over persisted pgvector chunks."""

from __future__ import annotations

import asyncio
import time

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse, SearchResult
from app.persistence.models import ChunkRow


def cosine_search_statement(query_vector: list[float], k: int) -> Select:
    """k-NN by cosine distance (``<=>``); matches a ``vector_cosine_ops`` HNSW index."""
    distance = ChunkRow.embedding.cosine_distance(query_vector)
    return (
        select(
            ChunkRow.id.label("chunk_id"),
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.metadata_json.label("metadata"),
            distance.label("distance"),
        )
        .where(ChunkRow.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )


async def search_chunks(
    request: SearchRequest,
    session: AsyncSession,
    embedder: OpenAIEmbedder,
) -> SearchResponse:
    """Embed the query and return its nearest persisted chunks by cosine distance."""
    started = time.perf_counter()
    query_vector = await asyncio.to_thread(embedder.embed_one, request.query)
    result = await session.execute(cosine_search_statement(query_vector, request.k))
    rows = result.mappings().all()

    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=round((time.perf_counter() - started) * 1000),
        results=[SearchResult.model_validate(row) for row in rows],
    )