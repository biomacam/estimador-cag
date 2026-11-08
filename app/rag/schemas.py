"""Pydantic contracts for the RAG estimation pipeline.

The ingest/search types (``Budget``, ``Chunk``, ``SearchRequest``...) stay in
``app.embedding_pipeline.schemas``; this module only holds the types of the
transcript -> grounded-estimate flow.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Scale = Literal["small", "medium", "large", "unknown"]
Confidence = Literal["high", "medium", "low", "insufficient"]
Relevance = Literal["primary", "supporting", "tangential"]
Impact = Literal["high", "medium", "low"]


class EstimationQuery(BaseModel):
    """Structured brief distilled from a raw meeting transcript.

    A transcript is full of digressions, so only what drives retrieval (what to
    build, with which tech, under which constraints) is extracted and embedded.
    """

    function: str = Field(description="Functional summary of the project.")
    technologies: list[str] = Field(default_factory=list)
    sector: str | None = None
    scale: Scale = "unknown"
    country: str | None = None
    regulations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """One chunk returned by the retriever; ``id`` is the chunk's DB primary key."""

    id: int
    content: str
    sector: str
    project_year: int
    chunk_type: str
    distance: float = Field(description="Cosine distance (lower = more similar).")
    budget_id: str | None = Field(
        default=None, description="Traceable corpus id of the parent budget (from JSONB metadata)."
    )


class RetrievalResult(BaseModel):
    """Outcome of the retrieval stage."""

    chunks: list[RetrievedChunk]
    low_confidence: bool = Field(
        description="True when no chunk crossed the distance threshold (soft-fail)."
    )
    candidates_evaluated: int = Field(
        ge=0, description="Total chunks scored before applying the threshold/limit."
    )


class SourceCitation(BaseModel):
    """A reference from the estimate back to a retrieved chunk."""

    source_id: int = Field(description="DB id of the cited chunk (a RetrievedChunk.id).")
    relevance: Relevance
    used_for: str = Field(description="What this source contributed to the estimate.")


class Assumption(BaseModel):
    """An estimate component NOT backed by any retrieved source."""

    description: str
    impact: Impact
    rationale: str


class TaskItem(BaseModel):
    """One concrete engineering task inside a functional module, in engineer-days."""

    name: str
    description: str | None = Field(default=None, description="One-line scope of the task.")
    engineer_days: int = Field(ge=0)
    sources: list[int] = Field(default_factory=list, description="Chunk ids that back this task.")


class WorkModule(BaseModel):
    """A functional block grouping the concrete tasks needed to deliver it."""

    name: str
    description: str | None = Field(default=None, description="What this functional block covers.")
    tasks: list[TaskItem] = Field(default_factory=list)


class Estimate(BaseModel):
    """Grounded estimate produced from retrieved historical budgets.

    Hours-based (engineer-days) with mandatory citations, distinct from the CAG
    ``EstimationResult`` (euros/weeks/phases). When the context is insufficient,
    ``confidence='insufficient'``, the totals stay ``None`` and ``modules`` is
    empty (enforced by ``validation.check_coherence``).
    """

    total_engineer_days: int | None = None
    modules: list[WorkModule] = Field(default_factory=list)
    duration_weeks: int | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="How the estimate was derived from the sources.")
    insufficient_context_explanation: str | None = None


# ---- HTTP request models ---------------------------------------------------


class RetrievalRequest(BaseModel):
    """Payload for ``POST /v1/retrieval/search`` (threshold + structural filters)."""

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0.0, le=2.0)
    sectors: list[str] | None = None
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = None
    # None = fall back to the runtime override, then the .env default.
    search_mode: Literal["vector", "hybrid"] | None = None
    rerank: bool | None = None


class EstimateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/from-transcript``."""

    transcript: str = Field(min_length=100, max_length=50_000)
    idempotency_key: str | None = Field(default=None, max_length=128)


# ---- Per-stage models: each pipeline step as a stateless endpoint ----------


class ReformulateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/reformulate``."""

    transcript: str = Field(min_length=100, max_length=50_000)


class ReformulationResult(BaseModel):
    """The structured brief plus the canonical search text that gets embedded."""

    query: EstimationQuery
    search_text: str = Field(description="Corpus-aligned text fed to the embedder.")


class AssembleRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/assemble``.

    ``max_context_tokens`` defaults to ``MAX_CONTEXT_TOKENS`` when omitted."""

    chunks: list[RetrievedChunk]
    max_context_tokens: int | None = Field(default=None, ge=256, le=64_000)


class AssembleResult(BaseModel):
    """The assembled ``<source>`` block plus what survived the token budget."""

    context_block: str
    kept_chunks: list[RetrievedChunk]
    dropped_count: int = Field(ge=0, description="Chunks dropped by the token budget.")
    token_count: int = Field(ge=0, description="Tokens in the assembled context block.")


class GenerateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/generate``.

    ``kept_chunks`` are the chunks the context block was built from; they are
    used to detect fabricated citations after generation."""

    context_block: str = Field(min_length=1)
    query: EstimationQuery
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)


class GenerateResult(BaseModel):
    """The estimate plus the grounding signals (no auto-retry in the stage)."""

    estimate: Estimate
    fabricated_source_ids: list[int] = Field(
        default_factory=list,
        description="Cited source ids not present in kept_chunks (empty = clean).",
    )
    coherent: bool = Field(description="False when an insufficient estimate still carries numbers.")
