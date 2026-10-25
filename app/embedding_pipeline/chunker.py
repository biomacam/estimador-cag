"""One chunk per budget component, retaining the parent project context.

The render helpers are shared by ``app.embedding_pipeline.strategies`` so every
strategy chunks the same textual representation of the corpus.
"""

from __future__ import annotations

import time

from app.embedding_pipeline.base import Chunker, count_tokens, emit_chunking_done
from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk


def render_component_text(budget: Budget, component: BudgetComponent) -> str:
    """Parent context header and component detail sent to the embedding API."""
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | "
        f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
        f"\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def serialize_budget(budget: Budget) -> str:
    """Whole budget as one text blob, for strategies that split full documents."""
    header = (
        f"[Project: {budget.project_summary}]\n"
        f"[Client: {budget.client_metadata.name} | "
        f"Sector: {budget.client_metadata.sector} | Year: {budget.year} | "
        f"Main tech: {budget.main_technology}]"
    )
    blocks = [header]
    for component in budget.components:
        blocks.append(
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
    return "\n\n".join(blocks)


def component_metadata(budget: Budget, component: BudgetComponent) -> dict:
    return {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }


class JSONStructuralChunker(Chunker):
    """Turns budgets into one chunk per component without splitting or overlap."""

    strategy_name = "structural"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                text = render_component_text(budget, component)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata=component_metadata(budget, component),
                        token_count=count_tokens(text),
                    )
                )
        emit_chunking_done(
            strategy=self.strategy_name,
            chunks=chunks,
            n_input_documents=len(budgets),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return chunks