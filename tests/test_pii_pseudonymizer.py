"""Tests for ConsistentPseudonymizer and the mapping stores.

Unit tests use a stub analyzer with canned results, so the substitution logic,
the consistency contract and the store integration are exercised without loading
spaCy. One integration test at the end runs the real Spanish analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.ingestion.pii import ConsistentPseudonymizer, InMemoryMappingStore, MappingStore


@dataclass
class _StubResult:
    """Minimal stand-in for presidio_analyzer.RecognizerResult."""

    entity_type: str
    start: int
    end: int
    score: float = 0.95


class StubAnalyzer:
    def __init__(self, results_by_text: dict[str, list[_StubResult]]) -> None:
        self._results = results_by_text

    def analyze(self, *, text: str, language: str, entities=None):
        return list(self._results.get(text, []))


def _span(text: str, needle: str, entity_type: str, score: float = 0.95) -> _StubResult:
    start = text.index(needle)
    return _StubResult(entity_type, start, start + len(needle), score)


def _pseudonymizer(analyzer, store=None, salt: str = "unit-test-salt") -> ConsistentPseudonymizer:
    return ConsistentPseudonymizer(
        analyzer=analyzer,
        mapping_store=store or InMemoryMappingStore(),
        salt=salt,
        faker_locale="es_ES",
        language="es",
    )


def test_same_value_gets_the_same_pseudonym_across_calls() -> None:
    text1 = "Habló con Laura Fernández ayer."
    text2 = "Laura Fernández firmó esta mañana."
    analyzer = StubAnalyzer(
        {
            text1: [_span(text1, "Laura Fernández", "PERSON")],
            text2: [_span(text2, "Laura Fernández", "PERSON")],
        }
    )
    pseudo = _pseudonymizer(analyzer)

    r1, r2 = pseudo.pseudonymize(text1), pseudo.pseudonymize(text2)

    assert r1.applied[0].pseudonym == r2.applied[0].pseudonym
    assert "Laura Fernández" not in r1.pseudonymized_text
    assert "Laura Fernández" not in r2.pseudonymized_text


def test_consistency_survives_a_new_pseudonymizer_sharing_the_store() -> None:
    text = "Habló con Laura Fernández ayer."
    analyzer = StubAnalyzer({text: [_span(text, "Laura Fernández", "PERSON")]})
    store = InMemoryMappingStore()

    first = _pseudonymizer(analyzer, store).pseudonymize(text).applied[0].pseudonym
    second = _pseudonymizer(analyzer, store).pseudonymize(text).applied[0].pseudonym

    assert first == second


def test_different_values_get_different_pseudonyms() -> None:
    text = "Laura Fernández y Javier Romero firmaron."
    analyzer = StubAnalyzer(
        {
            text: [
                _span(text, "Laura Fernández", "PERSON"),
                _span(text, "Javier Romero", "PERSON"),
            ]
        }
    )

    result = _pseudonymizer(analyzer).pseudonymize(text)

    assert len({m.pseudonym for m in result.applied}) == 2


def test_hash_depends_on_the_salt_and_never_contains_the_value() -> None:
    text = "Habló con Laura Fernández ayer."
    analyzer = StubAnalyzer({text: [_span(text, "Laura Fernández", "PERSON")]})

    a = _pseudonymizer(analyzer, salt="salt-a").pseudonymize(text).applied[0].original_hash
    b = _pseudonymizer(analyzer, salt="salt-b").pseudonymize(text).applied[0].original_hash

    assert a != b
    assert len(a) == 64
    assert "Laura" not in a


def test_budget_id_gets_a_budget_shaped_pseudonym() -> None:
    text = "Revisando BUDGET-2024-0001 con el cliente."
    analyzer = StubAnalyzer({text: [_span(text, "BUDGET-2024-0001", "BUDGET_ID")]})

    result = _pseudonymizer(analyzer).pseudonymize(text)

    assert result.applied[0].entity_type == "BUDGET_ID"
    assert result.applied[0].pseudonym.startswith("BUDGET-")
    assert "BUDGET-2024-0001" not in result.pseudonymized_text


def test_pseudonym_generation_is_reproducible_for_a_fresh_instance() -> None:
    text = "Cliente CLI-0042 activo."
    analyzer = StubAnalyzer({text: [_span(text, "CLI-0042", "CLIENT_CODE")]})

    first = _pseudonymizer(analyzer).pseudonymize(text).applied[0].pseudonym
    second = _pseudonymizer(analyzer).pseudonymize(text).applied[0].pseudonym

    assert first == second
    assert first.startswith("CLI-")


def test_unknown_entity_type_gets_a_fallback_pseudonym() -> None:
    text = "Su IBAN es ES9121000418450200051332 ok."
    analyzer = StubAnalyzer({text: [_span(text, "ES9121000418450200051332", "IBAN_CODE")]})

    result = _pseudonymizer(analyzer).pseudonymize(text)

    assert result.applied[0].pseudonym.startswith("REDACTED-")
    assert "ES9121000418450200051332" not in result.pseudonymized_text


def test_adjacent_entities_are_replaced_right_to_left() -> None:
    text = "Carmen escribió a carmen.vidal@toledo-ind.es y firmó."
    analyzer = StubAnalyzer(
        {
            text: [
                _span(text, "Carmen", "PERSON"),
                _span(text, "carmen.vidal@toledo-ind.es", "EMAIL_ADDRESS"),
            ]
        }
    )

    result = _pseudonymizer(analyzer).pseudonymize(text)

    assert "Carmen" not in result.pseudonymized_text.split()
    assert "carmen.vidal@toledo-ind.es" not in result.pseudonymized_text
    assert result.pseudonymized_text.endswith(" y firmó.")


def test_overlapping_detections_keep_the_widest_span() -> None:
    """A PERSON found inside an email must not corrupt the email replacement."""
    text = "Escribir a carmen.vidal@toledo-ind.es hoy."
    analyzer = StubAnalyzer(
        {
            text: [
                _span(text, "carmen", "PERSON", score=0.85),
                _span(text, "carmen.vidal@toledo-ind.es", "EMAIL_ADDRESS", score=1.0),
            ]
        }
    )

    result = _pseudonymizer(analyzer).pseudonymize(text)

    assert [m.entity_type for m in result.applied] == ["EMAIL_ADDRESS"]
    assert result.pseudonymized_text.startswith("Escribir a ")
    assert result.pseudonymized_text.endswith(" hoy.")
    assert "carmen" not in result.pseudonymized_text.lower().replace("carmen.vidal", "")


def test_no_pii_means_no_changes() -> None:
    text = "Una frase sin nada interesante."
    result = _pseudonymizer(StubAnalyzer({text: []})).pseudonymize(text)

    assert result.pseudonymized_text == text
    assert result.applied == []


def test_in_memory_store_forget_breaks_the_mapping() -> None:
    store = InMemoryMappingStore()
    assert isinstance(store, MappingStore)

    first = store.lookup_or_create("PERSON", "h", lambda: "A")
    assert store.forget("PERSON", "h") is True
    assert store.forget("PERSON", "h") is False
    assert store.lookup_or_create("PERSON", "h", lambda: "B") == "B"
    assert first == "A"


# --- real analyzer ----------------------------------------------------------


@pytest.fixture(scope="module")
def real_analyzer():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("es_core_news_md"):
        pytest.skip("es_core_news_md is not installed")
    from app.ingestion.pii.analyzer import build_analyzer

    return build_analyzer()


def test_real_analyzer_detects_spanish_names_and_domain_ids(real_analyzer) -> None:
    text = "Laura Fernández revisó BUDGET-2024-0001 del cliente CLI-0042."
    pseudo = ConsistentPseudonymizer(
        analyzer=real_analyzer,
        mapping_store=InMemoryMappingStore(),
        salt="unit-test-salt",
    )

    result = pseudo.pseudonymize(text)

    found = {m.entity_type for m in result.applied}
    assert {"PERSON", "BUDGET_ID", "CLIENT_CODE"} <= found
    for original in ("Laura Fernández", "BUDGET-2024-0001", "CLI-0042"):
        assert original not in result.pseudonymized_text
