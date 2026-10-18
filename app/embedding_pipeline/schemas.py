"""Input budgets, embeddable chunks and ingestion response models."""

from __future__ import annotations

from typing import Literal

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

    budgets: list[Budget] = Field(min_length=1, description="Budgets to chunk and embed.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"budgets": [{
                "budget_id": "BUD-2024-001",
                "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
                "project_summary": "Mobile banking API with OAuth 2.0 authentication and PSD2 compliance",
                "main_technology": "ruby_on_rails",
                "year": 2024,
                "total_estimated_hours": 480,
                "components": [
                    {
                        "component_id": "AUTH-001",
                        "name": "OAuth 2.0 authentication backend",
                        "description": "Implementation of OAuth 2.0 flows (authorization code, refresh token) with JWT-based session management, multi-tenant token isolation, and rate limiting per client.",
                        "tech_stack": ["ruby_on_rails", "postgresql", "redis"],
                        "estimated_hours": 120,
                        "complexity": "high",
                        "dependencies": [],
                    },
                    {
                        "component_id": "PSD2-002",
                        "name": "PSD2 open banking connectors",
                        "description": "Account information and payment initiation endpoints compliant with PSD2/Berlin Group, including strong customer authentication (SCA) and consent management.",
                        "tech_stack": ["ruby_on_rails", "postgresql"],
                        "estimated_hours": 160,
                        "complexity": "high",
                        "dependencies": ["AUTH-001"],
                    },
                    {
                        "component_id": "TXN-003",
                        "name": "Transaction ledger service",
                        "description": "Double-entry transaction ledger with idempotent writes, daily reconciliation jobs and an audit trail of every balance change.",
                        "tech_stack": ["ruby_on_rails", "postgresql", "sidekiq"],
                        "estimated_hours": 140,
                        "complexity": "medium",
                        "dependencies": ["AUTH-001"],
                    },
                    {
                        "component_id": "NOTIF-004",
                        "name": "Push and email notifications",
                        "description": "Event-driven notifications for transaction alerts and security events delivered via APNs/FCM and transactional email.",
                        "tech_stack": ["ruby_on_rails", "redis"],
                        "estimated_hours": 60,
                        "complexity": "low",
                        "dependencies": ["TXN-003"],
                    },
                ],
            }]}],
        },
    }


class IngestStats(BaseModel):
    """Aggregate ingestion counters."""

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


class IngestResponse(BaseModel):
    """Embedded chunks and aggregate stats."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats