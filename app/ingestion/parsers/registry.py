"""Parser registry: maps a format string to a :class:`Parser` instance.

A class rather than a bare dict so the "one parser per format" invariant is
enforced and the orchestrator has a single object to depend on.
"""

from __future__ import annotations

from app.ingestion.parsers.budget_json import BudgetJsonParser
from app.ingestion.parsers.protocol import Parser
from app.ingestion.parsers.transcript_txt import TranscriptTxtParser


class ParserRegistry:
    def __init__(self) -> None:
        self._by_format: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        for fmt in parser.supported_formats:
            if fmt in self._by_format:
                raise ValueError(
                    f"Format {fmt!r} already has a registered parser "
                    f"({type(self._by_format[fmt]).__name__})"
                )
            self._by_format[fmt] = parser

    def get(self, fmt: str) -> Parser:
        try:
            return self._by_format[fmt]
        except KeyError as exc:
            raise KeyError(f"No parser registered for format {fmt!r}") from exc

    def formats(self) -> set[str]:
        return set(self._by_format)


def default_registry() -> ParserRegistry:
    """Registry preloaded with the available parsers (JSON budgets, TXT transcripts).

    XLSX, DOCX and PDF parsers are not registered; adding one later is a single
    ``registry.register(...)`` call.
    """
    registry = ParserRegistry()
    registry.register(BudgetJsonParser())
    registry.register(TranscriptTxtParser())
    return registry
