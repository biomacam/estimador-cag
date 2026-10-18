"""Persistence layer for the ingestion subsystem.

Holds the SQLAlchemy engine, the declarative ``Base`` and the row models behind
the pseudonym mapping table and the ingestion job tracker. Higher layers use
the narrow repositories in ``app.persistence.repositories`` and never see
SQLAlchemy types.
"""

from app.persistence.database import create_engine_from_settings, get_session, new_session
from app.persistence.models import Base, IngestionJobRow, PseudonymMappingRow

__all__ = [
    "Base",
    "IngestionJobRow",
    "PseudonymMappingRow",
    "create_engine_from_settings",
    "get_session",
    "new_session",
]
