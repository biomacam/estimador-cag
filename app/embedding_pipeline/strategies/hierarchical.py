"""Hierarchical (parent-child) chunker.

Indexes two levels: children are components (precise, good recall) and the
parent is the whole serialized budget (full context for the generator). Each
child carries ``metadata.parent_chunk_id``.
"""

from __future__ import annotations

import time

from app.embedding_pipeline.base import Chunker, count_tokens, emit_chunking_done
from app.embedding_pipeline.chunker import (
    component_metadata,
    render_component_text,
    serialize_budget,
)
from app.embedding_pipeline.schemas import Budget, Chunk


class HierarchicalChunker(Chunker):
    strategy_name = "hierarchical"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for budget in budgets:
            parent_id = f"{budget.budget_id}::parent"
            parent_text = serialize_budget(budget)
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    metadata={
                        "budget_id": budget.budget_id,
                        "client_sector": budget.client_metadata.sector,
                        "main_technology": budget.main_technology,
                        "year": budget.year,
                        "level": "parent",
                    },
                    token_count=count_tokens(parent_text),
                )
            )
            for component in budget.components:
                child_text = render_component_text(budget, component)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=child_text,
                        metadata={
                            **component_metadata(budget, component),
                            "level": "child",
                            "parent_chunk_id": parent_id,
                        },
                        token_count=count_tokens(child_text),
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
