"""Persistence layer: lazy engine and the jobs repository (SQLite in-memory)."""

from __future__ import annotations

import subprocess
import sys
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.persistence.models import Base
from app.persistence.repositories.jobs import JobsRepository, list_jobs


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as db_session:
        yield db_session


def test_importing_persistence_does_not_build_an_engine() -> None:
    # Fresh interpreter: this process may already have built an engine in another test.
    code = (
        "import app.persistence as p; "
        "from app.persistence import database as d; "
        "assert d.create_engine_from_settings.cache_info().currsize == 0"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_job_lifecycle_completed(session: Session) -> None:
    repo = JobsRepository(session)

    job = repo.create("presupuestos_json")
    assert job.status == "pending"
    assert job.documents_count == 0

    repo.mark_running(job.job_id)
    assert repo.get(job.job_id).status == "running"

    repo.mark_completed(job.job_id, documents_count=6)
    done = repo.get(job.job_id)
    assert done.status == "completed"
    assert done.documents_count == 6
    assert done.finished_at is not None


def test_job_lifecycle_failed_truncates_the_message(session: Session) -> None:
    repo = JobsRepository(session)
    job = repo.create("presupuestos_json")

    repo.mark_failed(job.job_id, error_message="x" * 5000)

    failed = repo.get(job.job_id)
    assert failed.status == "failed"
    assert len(failed.error_message) == 2048
    assert failed.finished_at is not None


def test_get_unknown_job_returns_none(session: Session) -> None:
    assert JobsRepository(session).get(uuid.uuid4()) is None


def test_updating_unknown_job_is_a_noop(session: Session) -> None:
    JobsRepository(session).mark_running(uuid.uuid4())


def test_list_jobs_returns_created_jobs(session: Session) -> None:
    repo = JobsRepository(session)
    repo.create("a")
    repo.create("b")

    assert {job.source_name for job in list_jobs(session)} == {"a", "b"}
