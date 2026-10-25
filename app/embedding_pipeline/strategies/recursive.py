"""Recursive character chunker — the reasonable low-cost default.

Splits each serialized budget on a hierarchy of separators (paragraph → line →
sentence → word), descending only when a piece is still too big.
"""

from __future__ import annotations

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embedding_pipeline.base import Chunker, count_tokens, emit_chunking_done
from app.embedding_pipeline.chunker import serialize_budget
from app.embedding_pipeline.schemas import Budget, Chunk

CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 80
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    strategy_name = "recursive"

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=OVERLAP_TOKENS,
            separators=SEPARATORS,
        )

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for budget in budgets:
            for i, piece in enumerate(self._splitter.split_text(serialize_budget(budget))):
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::r{i}",
                        text=piece,
                        metadata={
                            "budget_id": budget.budget_id,
                            "client_sector": budget.client_metadata.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "part": i,
                        },
                        token_count=count_tokens(piece),
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
