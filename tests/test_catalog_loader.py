"""Tests for the YAML catalog loader, Pydantic validators and folder inspector."""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from app.ingestion.catalog import CatalogDecision, load_catalog
from app.ingestion.catalog.inspect import inspect_folder, inspect_root
from app.ingestion.catalog.models import (
    CatalogSource,
    DataCatalog,
    QualityScore,
    Sensitivity,
)

_QUALITY = {"completeness": 3, "consistency": 3, "actuality": 3, "reliability": 3}


def _source(name: str, **overrides: object) -> dict:
    return {
        "name": name,
        "location": "x",
        "format": "json",
        "quality": _QUALITY,
        "sensitivity": {"has_pii": False, "pii_flags": []},
        "decision": "include",
        **overrides,
    }


def _write_catalog(tmp_path, body: str):
    path = tmp_path / "catalog.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


_VALID_BODY = """
version: "1.0.0"
sources:
  - name: budgets_json
    location: budgets
    format: json
    quality: {completeness: 4, consistency: 3, actuality: 5, reliability: 4}
    sensitivity: {has_pii: true, pii_flags: ["PERSON"]}
    decision: include
  - name: legacy_txt
    location: transcripts
    format: txt
    quality: {completeness: 3, consistency: 2, actuality: 4, reliability: 3}
    sensitivity: {has_pii: true, pii_flags: ["PERSON"]}
    decision: review
    decision_reason: "legacy format pending owner sign-off"
"""


def test_load_catalog_happy_path(tmp_path) -> None:
    catalog = load_catalog(_write_catalog(tmp_path, _VALID_BODY))

    assert isinstance(catalog, DataCatalog)
    assert catalog.version == "1.0.0"
    assert len(catalog.sources) == 2
    assert [s.name for s in catalog.included_sources()] == ["budgets_json"]


def test_shipped_catalog_validates() -> None:
    catalog = load_catalog("data/catalog/catalog.yaml")

    assert catalog.included_sources(), "the shipped catalog needs at least one include"


def test_review_or_exclude_requires_reason(tmp_path) -> None:
    body = _VALID_BODY.replace('decision_reason: "legacy format pending owner sign-off"', "")

    with pytest.raises(ValidationError, match="decision_reason"):
        load_catalog(_write_catalog(tmp_path, body))


def test_duplicate_source_names_rejected() -> None:
    payload = {"version": "1.0.0", "sources": [_source("dup"), _source("dup")]}

    with pytest.raises(ValidationError, match="Duplicate source name"):
        DataCatalog.model_validate(payload)


def test_name_must_be_snake_case() -> None:
    with pytest.raises(ValidationError, match="snake_case"):
        CatalogSource(
            name="BudgetJSON",
            location="x",
            format="json",
            quality=QualityScore(**_QUALITY),
            sensitivity=Sensitivity(has_pii=False, pii_flags=[]),
            decision=CatalogDecision.INCLUDE,
        )


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        DataCatalog.model_validate(
            {"version": "1.0.0", "sources": [_source("a", unexpected="boom")]}
        )


def test_quality_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        QualityScore(completeness=6, consistency=3, actuality=3, reliability=3)


def test_find_returns_source_or_none(tmp_path) -> None:
    catalog = load_catalog(_write_catalog(tmp_path, _VALID_BODY))

    assert catalog.find("budgets_json") is not None
    assert catalog.find("does_not_exist") is None


def test_inspect_folder_reports_facts(tmp_path) -> None:
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.TXT").write_text("hi", encoding="utf-8")

    facts = inspect_folder(tmp_path)

    assert facts.file_count == 2
    assert facts.formats_detected == {"json", "txt"}
    assert facts.latest_modified is not None


def test_inspect_root_groups_children_and_direct_files(tmp_path) -> None:
    (tmp_path / "budgets").mkdir()
    (tmp_path / "budgets" / "one.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rates.xlsx").write_bytes(b"x")

    results = inspect_root(tmp_path)

    assert [r.file_count for r in results] == [1, 1]
    assert results[1].formats_detected == {"xlsx"}


def test_inspect_missing_folder_is_empty(tmp_path) -> None:
    assert inspect_folder(tmp_path / "nope").file_count == 0
    assert inspect_root(tmp_path / "nope") == []
