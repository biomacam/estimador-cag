"""One chunk per budget component, retaining the parent project context."""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
_ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)


class JSONStructuralChunker:
    """Turns budgets into one chunk per component without splitting or overlap."""

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._chunk_component(budget, component))
        return chunks

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": budget.client_metadata.sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=len(_ENCODING.encode(text)),
        )

    @staticmethod
    def _render_text(budget: Budget, component: BudgetComponent) -> str:
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