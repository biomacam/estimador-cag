"""The canonical ``Document`` model.

A single ``DocumentMetadata`` sub-object holds the structured fields every
consumer needs; anything format-specific lives in ``extra``. That escape hatch
keeps the contract uniform across JSON, TXT and future formats without a
sub-type per source.

The ``id`` is assigned by the parser and must be stable across re-ingestions so
a downstream index can dedupe.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_version: str  # the ``DataCatalog.version`` active at ingestion time
    ingested_at: datetime
    lineage: list[str] = Field(default_factory=list)
    sensitivity_pii_flags: list[str] = Field(default_factory=list)
    sensitivity_access_level: str = "internal"
    location: str = ""  # path or URL the document came from
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    metadata: DocumentMetadata
