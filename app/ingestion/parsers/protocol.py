"""The :class:`Parser` protocol and the :class:`ParseContext` passed to it.

A Protocol rather than an abstract base class: parsers share no implementation,
and structural typing lets tests drop in a fake without inheriting anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Protocol, runtime_checkable

from app.ingestion.catalog.models import CatalogSource
from app.ingestion.documents.models import Document
from app.ingestion.loaders.filesystem import LoadedBlob


@dataclass(frozen=True)
class ParseContext:
    """Everything a parser needs that does not live in the blob itself.

    Built once per ingestion run and reused across blobs, so ``ingested_at`` and
    ``source_version`` are constant for the whole run; downstream dedupe by hash
    relies on that.
    """

    source: CatalogSource
    source_version: str
    ingested_at: datetime


@runtime_checkable
class Parser(Protocol):
    supported_formats: ClassVar[set[str]]

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:  # pragma: no cover
        ...
