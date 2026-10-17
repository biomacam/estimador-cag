"""Tests for the ingestion orchestrator (SQLite-backed job tracking)."""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ingestion.catalog.models import DataCatalog
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.orchestrator import IngestionRejected, ingest_source
from app.ingestion.parsers import ParserRegistry, default_registry
from app.persistence.models import Base
from app.persistence.repositories.jobs import JobsRepository

_QUALITY = {"completeness": 4, "consistency": 4, "actuality": 4, "reliability": 4}


def _source(name: str, location: str, fmt: str, decision: str = "include") -> dict:
    return {
        "name": name,
        "location": location,
        "format": fmt,
        "quality": _QUALITY,
        "sensitivity": {"has_pii": True, "pii_flags": ["PERSON"]},
        "decision": decision,
        **({"decision_reason": "not ready"} if decision != "include" else {}),
    }


@pytest.fixture
def catalog() -> DataCatalog:
    return DataCatalog.model_validate(
        {
            "version": "2.0.0",
            "sources": [
                _source("budgets", "budgets", "json"),
                _source("transcripts", "transcripts", "txt", decision="review"),
                _source("rates", "rates", "xlsx"),
                _source("missing", "nowhere", "json"),
            ],
        }
    )


@pytest.fixture
def data_root(tmp_path):
    (tmp_path / "budgets").mkdir()
    for n in (1, 2):
        payload = {"budget_id": f"BUDGET-2024-000{n}", "currency": "EUR", "total_amount": n}
        (tmp_path / "budgets" / f"b{n}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "budgets" / "ignored.txt").write_text("not a budget", encoding="utf-8")
    (tmp_path / "rates").mkdir()
    (tmp_path / "rates" / "r.xlsx").write_bytes(b"x")
    return tmp_path


@pytest.fixture
def repo() -> JobsRepository:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session: Session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield JobsRepository(session)
    session.close()


def _run(catalog, data_root, repo, source_name, registry=None):
    job = repo.create(source_name)
    kwargs = dict(
        catalog=catalog,
        source_name=source_name,
        loader=FileSystemLoader(data_root),
        registry=registry or default_registry(),
        jobs_repo=repo,
        job_id=job.job_id,
    )
    return job, kwargs


def test_successful_run_returns_documents_and_completes_the_job(catalog, data_root, repo) -> None:
    job, kwargs = _run(catalog, data_root, repo, "budgets")

    documents = ingest_source(**kwargs)

    assert len(documents) == 2
    assert {d.metadata.source_version for d in documents} == {"2.0.0"}
    done = repo.get(job.job_id)
    assert done.status == "completed"
    assert done.documents_count == 2
    assert done.finished_at is not None


def test_run_uses_one_ingested_at_for_every_document(catalog, data_root, repo) -> None:
    _, kwargs = _run(catalog, data_root, repo, "budgets")

    documents = ingest_source(**kwargs)

    assert len({d.metadata.ingested_at for d in documents}) == 1


def test_unknown_source_is_rejected_without_touching_the_job(catalog, data_root, repo) -> None:
    job, kwargs = _run(catalog, data_root, repo, "nope")

    with pytest.raises(IngestionRejected, match="not found"):
        ingest_source(**kwargs)

    assert repo.get(job.job_id).status == "pending"


def test_non_included_source_is_rejected(catalog, data_root, repo) -> None:
    job, kwargs = _run(catalog, data_root, repo, "transcripts")

    with pytest.raises(IngestionRejected, match="review"):
        ingest_source(**kwargs)

    assert repo.get(job.job_id).status == "pending"


def test_missing_location_marks_the_job_failed(catalog, data_root, repo) -> None:
    job, kwargs = _run(catalog, data_root, repo, "missing")

    with pytest.raises(FileNotFoundError):
        ingest_source(**kwargs)

    failed = repo.get(job.job_id)
    assert failed.status == "failed"
    # Only the exception type is stored; the path stays out of the HTTP-visible job.
    assert failed.error_message == "FileNotFoundError"


def test_missing_parser_marks_the_job_failed_instead_of_running_forever(
    catalog, data_root, repo
) -> None:
    job, kwargs = _run(catalog, data_root, repo, "rates")

    with pytest.raises(KeyError, match="xlsx"):
        ingest_source(**kwargs)

    assert repo.get(job.job_id).status == "failed"


def test_a_corrupt_file_marks_the_job_failed(catalog, data_root, repo) -> None:
    (data_root / "budgets" / "bad.json").write_text("{not json", encoding="utf-8")
    job, kwargs = _run(catalog, data_root, repo, "budgets")

    with pytest.raises(json.JSONDecodeError):
        ingest_source(**kwargs)

    assert repo.get(job.job_id).status == "failed"


def test_an_empty_registry_fails_the_job(catalog, data_root, repo) -> None:
    job, kwargs = _run(catalog, data_root, repo, "budgets", registry=ParserRegistry())

    with pytest.raises(KeyError):
        ingest_source(**kwargs)

    assert repo.get(job.job_id).status == "failed"


def test_unknown_job_id_does_not_break_the_run(catalog, data_root, repo) -> None:
    _, kwargs = _run(catalog, data_root, repo, "budgets")
    kwargs["job_id"] = uuid.uuid4()

    assert len(ingest_source(**kwargs)) == 2
