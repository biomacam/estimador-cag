"""Parsers: bytes -> ``Document``.

:class:`Parser` is a :class:`typing.Protocol`, not a base class: a parser is
anything that exposes ``supported_formats`` and ``parse``. Parsers share no
state; they are pure functions over bytes.
"""

from app.ingestion.parsers.budget_json import BudgetJsonParser
from app.ingestion.parsers.protocol import Parser, ParseContext
from app.ingestion.parsers.registry import ParserRegistry, default_registry
from app.ingestion.parsers.transcript_txt import TranscriptTxtParser

__all__ = [
    "BudgetJsonParser",
    "Parser",
    "ParseContext",
    "ParserRegistry",
    "TranscriptTxtParser",
    "default_registry",
]
