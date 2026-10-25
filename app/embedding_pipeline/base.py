"""Common interface for chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
import tiktoken

from app.embedding_pipeline.schemas import Budget, Chunk

log = structlog.get_logger()

# cl100k_base is the tokenizer of text-embedding-3-small, so stats stay comparable.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def emit_chunking_done(
    *,
    strategy: str,
    chunks: list[Chunk],
    n_input_documents: int,
    extra_api_calls: int = 0,
    extra_cost_usd: float = 0.0,
    latency_ms: float = 0.0,
) -> None:
    log.info(
        "chunking_done",
        strategy=strategy,
        n_chunks=len(chunks),
        n_input_documents=n_input_documents,
        extra_api_calls=extra_api_calls,
        extra_cost_usd=round(extra_cost_usd, 6),
        latency_ms=round(latency_ms, 1),
    )


class Chunker(ABC):
    """A chunking strategy: budgets in, chunks out.

    Subclasses update ``last_extra_api_calls`` / ``last_extra_cost_usd`` after each
    ``chunk()`` call; strategies without external API calls leave them at zero.
    """

    last_extra_api_calls: int = 0
    last_extra_cost_usd: float = 0.0

    @abstractmethod
    def chunk(self, budgets: list[Budget]) -> list[Chunk]: ...

    @property
    @abstractmethod
    def strategy_name(self) -> str: ...
