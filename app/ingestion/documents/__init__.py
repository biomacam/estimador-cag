"""The canonical ``Document`` contract.

Every parser produces ``Document`` instances and every downstream consumer
(cleaning, validation, PII, future RAG) accepts them. The model is deliberately
flat (no nested business types) so it is trivial to fake in tests and
homogeneous across formats.
"""

from app.ingestion.documents.models import Document, DocumentMetadata

__all__ = ["Document", "DocumentMetadata"]
