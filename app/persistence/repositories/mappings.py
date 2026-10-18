"""Pseudonym mapping repository.

``lookup_or_create`` must be idempotent under contention: two parallel calls
with the same ``(entity_type, original_hash)`` must converge to the same
pseudonym, otherwise one person would get two pseudonyms in the corpus. The
UNIQUE constraint enforces it: insert first, fall back to a SELECT on conflict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.persistence.models import PseudonymMappingRow


@dataclass(frozen=True)
class Mapping:
    entity_type: str
    original_hash: str
    pseudonym: str
    created_at: datetime


class MappingsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def lookup(self, entity_type: str, original_hash: str) -> Mapping | None:
        row = self._session.execute(
            select(PseudonymMappingRow).where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        ).scalar_one_or_none()
        return _to_mapping(row) if row is not None else None

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> Mapping:
        """Return the existing mapping or insert a new one; the factory runs only on a miss."""
        existing = self.lookup(entity_type, original_hash)
        if existing is not None:
            return existing

        stmt = (
            pg_insert(PseudonymMappingRow)
            .values(
                entity_type=entity_type,
                original_hash=original_hash,
                pseudonym=new_pseudonym_factory(),
            )
            .on_conflict_do_nothing(index_elements=["entity_type", "original_hash"])
            .returning(PseudonymMappingRow)
        )
        row = self._session.execute(stmt).scalar_one_or_none()
        self._session.commit()

        if row is not None:
            return _to_mapping(row)

        # Another writer won the race; read the winning row.
        winner = self.lookup(entity_type, original_hash)
        assert winner is not None, "unique constraint guarantees a winner exists"
        return winner

    def forget(self, entity_type: str, original_hash: str) -> bool:
        """Delete a mapping. Returns True if a row existed."""
        deleted = self._session.execute(
            PseudonymMappingRow.__table__.delete().where(
                PseudonymMappingRow.entity_type == entity_type,
                PseudonymMappingRow.original_hash == original_hash,
            )
        )
        self._session.commit()
        return deleted.rowcount > 0


def _to_mapping(row: PseudonymMappingRow) -> Mapping:
    return Mapping(
        entity_type=row.entity_type,
        original_hash=row.original_hash,
        pseudonym=row.pseudonym,
        created_at=row.created_at,
    )
