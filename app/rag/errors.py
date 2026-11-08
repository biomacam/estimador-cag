"""Exception hierarchy for the RAG estimation pipeline.

Each stage raises a specific subclass so the API layer can map failures to the
right HTTP status. Never raise a bare ``Exception`` from pipeline code.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class for every error raised inside the RAG pipeline."""


class ReformulationError(RagError):
    """Query understanding failed even after the simple-rewrite fallback."""


class RetrievalError(RagError):
    """Retrieval or cross-encoder reranking failed."""


class GenerationError(RagError):
    """The LLM generation step failed irrecoverably."""


class CitationValidationError(RagError):
    """The model kept citing fabricated source ids after the retry budget."""


class MalformedEstimateError(RagError):
    """The generated estimate violates the insufficient-context coherence rule
    even after one repair attempt."""
