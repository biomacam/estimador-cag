"""Narrow repositories over the persistence layer.

Each wraps a single ``Session`` and exposes only the operations its caller
needs; no ORM types leak past this boundary.
"""

from app.persistence.repositories.jobs import JobsRepository
from app.persistence.repositories.mappings import MappingsRepository

__all__ = ["JobsRepository", "MappingsRepository"]
