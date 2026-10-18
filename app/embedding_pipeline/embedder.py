"""Batched OpenAI embeddings with rate-limit retries and cost estimation."""

from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100
PRICE_PER_MILLION_TOKENS_USD = 0.02
"""Time-sensitive OpenAI input pricing: USD per million tokens; update when pricing changes."""
_RETRY_BACKOFF_SECONDS = (1, 2, 4)


def estimated_cost_usd(total_tokens: int) -> float:
    """Estimated input cost at $0.02 per million tokens for the default model."""
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    """Wrap the embeddings API with batching and exponential backoff."""

    def __init__(self, client: OpenAI, model: str = MODEL) -> None:
        self._client = client
        self._model = model

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text for the comparison CLI."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed every chunk in order, batching API calls."""
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [chunk.text for chunk in batch]
            batch_tokens = sum(chunk.token_count for chunk in batch)
            started = time.perf_counter()
            vectors = self._create(texts)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            log.info(
                "embedding_batch_done",
                chunks=len(batch),
                tokens=batch_tokens,
                latency_ms=latency_ms,
                model=self._model,
            )
            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(**chunk.model_dump(), embedding=vector))
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        """Retry rate limits; propagate other provider errors."""
        last_error: RateLimitError | None = None
        for wait in (0, *_RETRY_BACKOFF_SECONDS):
            if wait:
                time.sleep(wait)
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except RateLimitError as exc:
                last_error = exc
                attempt = (0, *_RETRY_BACKOFF_SECONDS).index(wait)
                if attempt < len(_RETRY_BACKOFF_SECONDS):
                    log.warning(
                        "embedding_rate_limited",
                        retry_in_s=_RETRY_BACKOFF_SECONDS[attempt],
                    )
                else:
                    log.warning("embedding_rate_limit_exhausted", attempts=attempt + 1)
        assert last_error is not None
        raise last_error