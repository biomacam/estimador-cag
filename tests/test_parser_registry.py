"""Tests for the parser registry."""

from __future__ import annotations

from typing import ClassVar

import pytest

from app.ingestion.parsers import (
    BudgetJsonParser,
    Parser,
    ParserRegistry,
    TranscriptTxtParser,
    default_registry,
)


def test_default_registry_has_json_and_txt() -> None:
    registry = default_registry()

    assert registry.formats() == {"json", "txt"}
    assert isinstance(registry.get("json"), BudgetJsonParser)
    assert isinstance(registry.get("txt"), TranscriptTxtParser)


def test_unregistered_format_raises() -> None:
    with pytest.raises(KeyError, match="xlsx"):
        default_registry().get("xlsx")


def test_one_parser_per_format() -> None:
    registry = ParserRegistry()
    registry.register(BudgetJsonParser())

    with pytest.raises(ValueError, match="already has a registered parser"):
        registry.register(BudgetJsonParser())


def test_a_plain_class_satisfies_the_parser_protocol() -> None:
    class FakeParser:
        supported_formats: ClassVar[set[str]] = {"csv"}

        def parse(self, blob, context):
            return []

    registry = ParserRegistry()
    registry.register(FakeParser())

    assert isinstance(FakeParser(), Parser)
    assert registry.formats() == {"csv"}
