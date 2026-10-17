"""The shipped catalog and seed data must work together end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ingestion.catalog import load_catalog
from app.ingestion.cleaning import clean_budget_records, validate_with_policy
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.orchestrator import ingest_source
from app.ingestion.parsers import default_registry
from app.ingestion.parsers.protocol import ParseContext
from app.ingestion.pii import ConsistentPseudonymizer, InMemoryMappingStore
from app.persistence.models import Base
from app.persistence.repositories.jobs import JobsRepository

CATALOG_PATH = Path("data/catalog/catalog.yaml")
SEED_ROOT = Path("data/seed")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(CATALOG_PATH)


@pytest.fixture
def jobs_repo():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield JobsRepository(session)
    session.close()


def _budget_records() -> list[dict]:
    paths = sorted((SEED_ROOT / "budgets").glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_every_catalog_location_exists_in_the_seed(catalog) -> None:
    for source in catalog.sources:
        assert (SEED_ROOT / source.location).exists(), source.name


def test_ingesting_the_included_source_yields_every_budget(catalog, jobs_repo) -> None:
    job = jobs_repo.create("presupuestos_json")

    documents = ingest_source(
        catalog=catalog,
        source_name="presupuestos_json",
        loader=FileSystemLoader(SEED_ROOT),
        registry=default_registry(),
        jobs_repo=jobs_repo,
        job_id=job.job_id,
    )

    assert len(documents) == 6
    assert len({d.id for d in documents}) == 6
    assert jobs_repo.get(job.job_id).documents_count == 6
    assert all(d.metadata.sensitivity_access_level == "confidential" for d in documents)


def test_transcripts_parse_in_both_modes(catalog) -> None:
    source = catalog.find("transcripciones_txt")
    context = ParseContext(source=source, source_version=catalog.version, ingested_at=pd.Timestamp.now(tz="UTC"))
    parser = default_registry().get("txt")

    modes = {
        blob.relative_path: {d.metadata.extra["format_mode"] for d in parser.parse(blob, context)}
        for blob in FileSystemLoader(SEED_ROOT).iter_blobs(source.location, {"txt"})
    }

    assert modes == {
        "transcripts/transcripcion_2023-09-08_legacy.txt": {"legacy"},
        "transcripts/transcripcion_2025-01-15_acme.txt": {"tagged"},
        "transcripts/transcripcion_2025-02-03_betanorte.txt": {"tagged"},
    }


def test_cleaning_the_seed_budgets_routes_each_one_correctly() -> None:
    df = clean_budget_records(_budget_records())

    # The two BUDGET-2024-0005 versions collapse into the latest one.
    assert len(df) == 5
    assert df.set_index("budget_id").loc["BUDGET-2024-0005", "total_amount"] == 32000.0
    # "12/03/2024" must survive next to ISO dates, and "eur" must be upper-cased.
    row = df.set_index("budget_id").loc["BUDGET-2024-0002"]
    assert row["signed_at"] == pd.Timestamp("2024-03-12")
    assert row["currency"] == "EUR"

    result = validate_with_policy(df)

    assert sorted(result.valid["budget_id"]) == [
        "BUDGET-2024-0001",
        "BUDGET-2024-0002",
        "BUDGET-2024-0003",
        "BUDGET-2024-0005",
    ]
    assert list(result.discarded["budget_id"]) == ["BUDGET-2024-0004"]
    assert result.report["failures_by_check"] == {"greater_than_or_equal_to(0)": 1}


def test_mixed_date_formats_do_not_depend_on_row_order() -> None:
    base = {"currency": "EUR", "client_name": "A", "client_code": "CLI-0001", "total_amount": 1.0}
    iso = {**base, "budget_id": "BUDGET-2024-0001", "signed_at": "2024-02-14"}
    spanish = {**base, "budget_id": "BUDGET-2024-0002", "signed_at": "12/03/2024"}

    forward = clean_budget_records([iso, spanish]).set_index("budget_id")["signed_at"]
    backward = clean_budget_records([spanish, iso]).set_index("budget_id")["signed_at"]

    assert forward.equals(backward)
    assert forward["BUDGET-2024-0002"] == pd.Timestamp("2024-03-12")


@pytest.fixture(scope="module")
def real_analyzer():
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("es_core_news_md"):
        pytest.skip("es_core_news_md is not installed")
    from app.ingestion.pii.analyzer import build_analyzer

    return build_analyzer()


def test_seed_budget_text_is_pseudonymized_consistently(catalog, jobs_repo, real_analyzer) -> None:
    job = jobs_repo.create("presupuestos_json")
    documents = ingest_source(
        catalog=catalog,
        source_name="presupuestos_json",
        loader=FileSystemLoader(SEED_ROOT),
        registry=default_registry(),
        jobs_repo=jobs_repo,
        job_id=job.job_id,
    )
    pseudonymizer = ConsistentPseudonymizer(
        analyzer=real_analyzer, mapping_store=InMemoryMappingStore(), salt="unit-test-salt"
    )

    results = [pseudonymizer.pseudonymize(d.text) for d in documents]
    output = "\n".join(r.pseudonymized_text for r in results)

    for leaked in ("laura.fernandez@acme.es", "carmen.vidal@toledo-ind.es", "CLI-0042", "CLI-0288"):
        assert leaked not in output
    # Carmen Vidal's email appears in both BUDGET-2024-0005 versions: one hash, one pseudonym.
    emails = [m for r in results for m in r.applied if m.entity_type == "EMAIL_ADDRESS"]
    pseudonyms_by_hash: dict[str, list[str]] = {}
    for m in emails:
        pseudonyms_by_hash.setdefault(m.original_hash, []).append(m.pseudonym)
    repeated = [p for p in pseudonyms_by_hash.values() if len(p) == 2]
    assert len(repeated) == 1
    assert len(set(repeated[0])) == 1
