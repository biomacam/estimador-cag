"""Sentence-window chunker.

Embeds single sentences (high recall) but carries a ±N-sentence window in
``metadata.window_text`` to feed the generator at retrieval time.
"""

from __future__ import annotations

import time

import nltk

from app.embedding_pipeline.base import Chunker, count_tokens, emit_chunking_done
from app.embedding_pipeline.chunker import component_metadata
from app.embedding_pipeline.schemas import Budget, Chunk

WINDOW_SIZE = 2


def _ensure_punkt() -> None:
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)


class SentenceWindowChunker(Chunker):
    strategy_name = "sentence_window"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        _ensure_punkt()
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                sentences = nltk.sent_tokenize(component.description)
                for i, sentence in enumerate(sentences):
                    lo = max(0, i - WINDOW_SIZE)
                    hi = min(len(sentences), i + WINDOW_SIZE + 1)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{component.component_id}::s{i}",
                            text=sentence,
                            metadata={
                                **component_metadata(budget, component),
                                "sentence_index": i,
                                "window_text": " ".join(sentences[lo:hi]),
                            },
                            token_count=count_tokens(sentence),
                        )
                    )
        self.last_extra_api_calls = 0
        self.last_extra_cost_usd = 0.0
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks
