"""Chunking comparison: per-strategy corpus stats and in-memory top-k cosine search.

Nothing is persisted. Each strategy is chunked once per ``budgets`` object so
stats and queries never double the LLM calls of the expensive strategies.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.embedding_pipeline.base import Chunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Budget, Chunk
from app.embedding_pipeline.similarity import cosine_similarity, percentile

ORPHAN_MAX_TOKENS = 20
OBESE_MIN_TOKENS = 800


class TokenDistribution(BaseModel):
    min: int = 0
    p50: float = 0.0
    p95: float = 0.0
    max: int = 0


class ChunkingStats(BaseModel):
    strategy: str
    n_chunks: int
    token_distribution: TokenDistribution
    n_orphan_chunks: int
    n_obese_chunks: int
    ingestion_cost_usd: float
    ingestion_seconds: float


class TopChunk(BaseModel):
    chunk_id: str
    cosine: float
    text_preview: str


class QueryResult(BaseModel):
    strategy: str
    query: str
    top_k: list[TopChunk] = Field(default_factory=list)


class CompareRequest(BaseModel):
    """Payload for POST /embeddings/compare."""

    budgets: list[Budget] = Field(min_length=1)
    queries: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(
        default_factory=list, description="Empty = every registered strategy."
    )
    top_k: int = Field(default=3, ge=1, le=10)


class CompareResponse(BaseModel):
    stats_per_strategy: dict[str, ChunkingStats]
    queries_per_strategy: dict[str, list[QueryResult]]


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class ChunkingComparator:
    """Runs a set of chunkers over the same corpus and compares them."""

    def __init__(self, chunkers: list[Chunker], embedder: OpenAIEmbedder) -> None:
        self._chunkers = chunkers
        self._embedder = embedder
        self._materialized: dict[str, dict] | None = None
        self._materialized_for: list[Budget] | None = None

    def _materialize(self, budgets: list[Budget]) -> dict[str, dict]:
        if self._materialized is not None and self._materialized_for is budgets:
            return self._materialized
        out: dict[str, dict] = {}
        for chunker in self._chunkers:
            t0 = time.perf_counter()
            chunks = chunker.chunk(budgets)
            out[chunker.strategy_name] = {
                "chunks": chunks,
                "elapsed": time.perf_counter() - t0,
                "cost": chunker.last_extra_cost_usd,
            }
        self._materialized = out
        self._materialized_for = budgets
        return out

    def compute_stats(self, budgets: list[Budget]) -> dict[str, ChunkingStats]:
        return {
            name: self._stats_for(name, data["chunks"], data["elapsed"], data["cost"])
            for name, data in self._materialize(budgets).items()
        }

    def run_queries(
        self, budgets: list[Budget], queries: list[str], top_k: int = 3
    ) -> dict[str, list[QueryResult]]:
        if not queries:
            return {}
        materialized = self._materialize(budgets)
        query_vectors = {q: self._embedder.embed_one(q) for q in queries}
        results: dict[str, list[QueryResult]] = {}
        for name, data in materialized.items():
            embedded = self._embedder.embed_many(data["chunks"])
            per_query: list[QueryResult] = []
            for query in queries:
                qvec = query_vectors[query]
                scored = sorted(
                    (
                        TopChunk(
                            chunk_id=c.chunk_id,
                            cosine=round(cosine_similarity(qvec, c.embedding), 4),
                            text_preview=_preview(c.text),
                        )
                        for c in embedded
                    ),
                    key=lambda t: t.cosine,
                    reverse=True,
                )
                per_query.append(QueryResult(strategy=name, query=query, top_k=scored[:top_k]))
            results[name] = per_query
        return results

    @staticmethod
    def _stats_for(name: str, chunks: list[Chunk], elapsed: float, cost: float) -> ChunkingStats:
        token_counts = [c.token_count for c in chunks]
        return ChunkingStats(
            strategy=name,
            n_chunks=len(chunks),
            token_distribution=TokenDistribution(
                min=min(token_counts) if token_counts else 0,
                p50=round(percentile(token_counts, 50), 1),
                p95=round(percentile(token_counts, 95), 1),
                max=max(token_counts) if token_counts else 0,
            ),
            n_orphan_chunks=sum(1 for t in token_counts if t < ORPHAN_MAX_TOKENS),
            n_obese_chunks=sum(1 for t in token_counts if t > OBESE_MIN_TOKENS),
            ingestion_cost_usd=round(cost, 6),
            ingestion_seconds=round(elapsed, 3),
        )
