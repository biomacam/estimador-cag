"""Input budgets, embeddable chunks and ingestion response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Client context carried alongside the budget."""

    name: str = Field(description="Client company name.")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(description="ISO-ish country code, e.g. 'ES'.")


class BudgetComponent(BaseModel):
    """A single line item of a historical budget."""

    component_id: str = Field(description="Stable id within the budget, e.g. 'AUTH-001'.")
    name: str = Field(description="Short human-readable component name.")
    description: str = Field(description="Detailed description of the work.")
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies involved in this component."
    )
    estimated_hours: int = Field(ge=0, description="Hours estimated for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_ids this one depends on."
    )


class Budget(BaseModel):
    """A complete historical budget with its components."""

    budget_id: str = Field(description="Stable budget id, e.g. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(description="One-line summary of the project.")
    main_technology: str = Field(description="Primary technology / stack of the project.")
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    total_estimated_hours: int = Field(ge=0, description="Sum of component hours, as recorded.")
    components: list[BudgetComponent] = Field(min_length=1, description="Budget line items.")


class Chunk(BaseModel):
    """Embeddable text with separate filterable metadata."""

    chunk_id: str = Field(description="Traceable id, format '{budget_id}::{component_id}'.")
    text: str = Field(description="Embeddable text: parent context + component detail.")
    metadata: dict = Field(default_factory=dict, description="Filterable, non-embedded fields.")
    token_count: int = Field(ge=0, description="Token count of text (tiktoken).")


class EmbeddedChunk(Chunk):
    """A chunk with its embedding vector attached."""

    embedding: list[float] = Field(
        description="Dense embedding vector (1536 dims for text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Payload for POST /embeddings/ingest."""

    source_path: str = Field(min_length=1, description="Stable source path for deduplication.")
    document_type: str = Field(min_length=1, max_length=50)
    content: Budget = Field(description="Complete source budget consumed by the structural chunker.")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "source_path": "data/budgets/budget_2024_q1_fintech.json",
                "document_type": "historical_budget",
                "content": {
                    "budget_id": "BUD-2024-001",
                    "client_metadata": {
                        "name": "FintechCorp", "sector": "finance", "country": "ES"
                    },
                    "project_summary": "Mobile banking API with OAuth 2.0 authentication",
                    "main_technology": "ruby_on_rails",
                    "year": 2024,
                    "total_estimated_hours": 120,
                    "components": [{
                        "component_id": "AUTH-001",
                        "name": "OAuth 2.0 authentication backend",
                        "description": "Implement OAuth 2.0 authentication flows.",
                        "tech_stack": ["ruby_on_rails", "postgresql"],
                        "estimated_hours": 120,
                        "complexity": "high",
                        "dependencies": [],
                    }],
                },
            }],
        },
    }


class IngestResponse(BaseModel):
    """Persisted document identifiers and ingestion metrics."""

    document_id: int
    chunks_created: int = Field(ge=0)
    embedding_dimension: int = Field(gt=0)
    ingestion_time_ms: int = Field(ge=0)


class IngestConflictResponse(BaseModel):
    """Conflict response when a source path has already been ingested."""

    detail: str
    document_id: int


class SearchRequest(BaseModel):
    """Query and result count for POST /search."""

    query: str = Field(min_length=1, max_length=8192)
    k: int = Field(ge=1, le=100)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    k: int
    search_time_ms: int = Field(ge=0)
    results: list[SearchResult]