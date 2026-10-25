"""Fixed-size chunker — the degenerate baseline.

Slides a fixed token window with overlap over each rendered component, ignoring
every natural boundary. Only useful as the floor for the smarter strategies.
"""

from __future__ import annotations

import time

import tiktoken

from app.embedding_pipeline.base import Chunker, emit_chunking_done
from app.embedding_pipeline.chunker import component_metadata, render_component_text
from app.embedding_pipeline.schemas import Budget, Chunk

CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 80

_ENCODING = tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker(Chunker):
    strategy_name = "fixed_size"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                tokens = _ENCODING.encode(render_component_text(budget, component))
                part = 0
                start = 0
                while start < len(tokens):
                    window = tokens[start : start + CHUNK_SIZE_TOKENS]
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{component.component_id}::p{part}",
                            text=_ENCODING.decode(window),
                            metadata={**component_metadata(budget, component), "part": part},
                            token_count=len(window),
                        )
                    )
                    part += 1
                    if start + CHUNK_SIZE_TOKENS >= len(tokens):
                        break
                    start += step
        self.last_extra_api_calls = 0
        self.last_extra_cost_usd = 0.0
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks
