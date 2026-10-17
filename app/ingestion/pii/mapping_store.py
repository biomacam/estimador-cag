"""The mapping store: persistent backing for consistent pseudonymization.

Two implementations:

* :class:`PostgresMappingStore`: the production path, wrapping
  :class:`MappingsRepository`. Only the HMAC of the original value reaches the
  database, never the value, so deleting a row by hash kills reversibility
  without a plaintext ever having been stored.

* :class:`InMemoryMappingStore`: for tests and one-off scripts, same contract.

Both are idempotent: looking up the same ``(entity_type, hash)`` twice returns
the same pseudonym regardless of call order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.persistence.repositories.mappings import MappingsRepository


@runtime_checkable
class MappingStore(Protocol):
    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:  # pragma: no cover
        ...

    def forget(self, entity_type: str, original_hash: str) -> bool:  # pragma: no cover
        ...


class PostgresMappingStore:
    """Persistent mapping store backed by Postgres."""

    def __init__(self, session: Session) -> None:
        self._repo = MappingsRepository(session)

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        mapping = self._repo.lookup_or_create(
            entity_type=entity_type,
            original_hash=original_hash,
            new_pseudonym_factory=new_pseudonym_factory,
        )
        return mapping.pseudonym

    def forget(self, entity_type: str, original_hash: str) -> bool:
        return self._repo.forget(entity_type, original_hash)


class InMemoryMappingStore:
    """Simple dict-backed mapping store for tests and scripts."""

    def __init__(self) -> None:
        self._mappings: dict[tuple[str, str], str] = {}

    def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> str:
        key = (entity_type, original_hash)
        if key not in self._mappings:
            self._mappings[key] = new_pseudonym_factory()
        return self._mappings[key]

    def forget(self, entity_type: str, original_hash: str) -> bool:
        return self._mappings.pop((entity_type, original_hash), None) is not None
