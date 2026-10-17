"""SQLAlchemy ORM models: pseudonym mappings and ingestion job tracking.

* ``pseudonym_mappings`` keys on ``(entity_type, original_hash)``. The hash is
  HMAC-SHA256 over the original value with a server-side salt, so a read of the
  DB alone cannot reconstruct the original; deleting a row by hash makes the
  mapping irreversible (GDPR Art. 17).

* ``ingestion_jobs`` backs the asynchronous ``POST /ingestion/runs`` endpoint: a
  row is created on request and a BackgroundTask moves it through
  ``running`` / ``completed`` / ``failed``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import DateTime


class Base(DeclarativeBase):
    """Single declarative base, picked up by Alembic's env.py."""


class PseudonymMappingRow(Base):
    __tablename__ = "pseudonym_mappings"
    __table_args__ = (
        UniqueConstraint("entity_type", "original_hash", name="uq_mappings_entity_hash"),
        Index("idx_mappings_lookup", "entity_type", "original_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    original_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    pseudonym: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IngestionJobRow(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("idx_jobs_status", "status"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    documents_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
