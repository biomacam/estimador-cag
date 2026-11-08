"""Cross-encoder reranker wrapper (recall-then-rerank).

A cross-encoder scores each ``(query, document)`` pair TOGETHER, which is far more
accurate than comparing two independently computed vectors but too slow to run over
the whole corpus: the retrieval branches recall a wide candidate set cheaply and
this rescoring keeps the best few. The recall width and final cut are decided by
``pipeline.retrieve``, not here.

The model loads lazily on first use so importing this module (tests, app startup)
never requires torch or weights on disk.
"""

from __future__ import annotations

import threading
import time
from math import isfinite

import structlog

from app.config import get_settings

log = structlog.get_logger()


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder that rescores ``(query, document)`` pairs."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None  # sentence_transformers.CrossEncoder, loaded on demand
        self._load_lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> "CrossEncoderReranker":
        return cls(get_settings().RERANKER_MODEL)

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        """Force-load the model now (warmup)."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:  # another thread won the race
                return self._model
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            log.info(
                "reranker_loaded",
                model=self._model_name,
                load_ms=int((time.perf_counter() - started) * 1000),
            )
            return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance of ``query`` against each document (higher = better), in input order."""
        if not documents:
            return []
        model = self._ensure_loaded()
        pairs = [(query, document) for document in documents]
        started = time.perf_counter()
        scores = [float(score) for score in model.predict(pairs)]
        if len(scores) != len(documents) or not all(isfinite(score) for score in scores):
            log.error(
                "reranker_invalid_scores", model=self._model_name,
                expected=len(documents), received=len(scores),
            )
            raise ValueError("Cross-encoder must return one finite score per document.")
        log.info(
            "reranker_scored",
            model=self._model_name,
            pairs=len(pairs),
            score_ms=int((time.perf_counter() - started) * 1000),
        )
        return scores

    def rerank(
        self,
        query: str,
        candidates: list,
        *,
        top_n: int,
        text_of=lambda candidate: candidate.content,
    ) -> list:
        """Reorder ``candidates`` by cross-encoder score and keep the top ``top_n``.

        The sort is stable, so equal scores keep their recall order.
        """
        if top_n <= 0:
            raise ValueError("Reranker top_n must be positive.")
        if not candidates:
            return []
        scores = self.score(query, [text_of(candidate) for candidate in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [candidate for candidate, _score in ranked[:top_n]]
