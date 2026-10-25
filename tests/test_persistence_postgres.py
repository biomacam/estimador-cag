"""Postgres-backed tests: migrations and the concurrent pseudonym mapping.

SQLite cannot run ``INSERT .. ON CONFLICT`` or the Postgres UUID type, so these need a
real server. They are skipped unless ``TEST_DATABASE_URL`` points to one, e.g.::

    docker run -d --rm -p 5434:5432 -e POSTGRES_USER=estimator \
        -e POSTGRES_PASSWORD=estimator -e POSTGRES_DB=estimator pgvector/pgvector:pg16
    $env:TEST_DATABASE_URL = "postgresql+psycopg://estimator:estimator@localhost:5434/estimator"

Each test module run creates its own throwaway database and drops it afterwards, so the
database named in the URL is never modified.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.ingestion import DuplicateDocumentError, ingest_document
from app.embedding_pipeline.schemas import Budget, EmbeddedChunk, IngestRequest, SearchRequest
from app.embedding_pipeline.search import search_chunks
from app.ingestion.pii import PostgresMappingStore
from app.persistence.models import ChunkRow, DocumentRow
from app.persistence.repositories.jobs import JobsRepository
from app.persistence.repositories.mappings import MappingsRepository

_ADMIN_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not _ADMIN_URL, reason="TEST_DATABASE_URL is not set")


def _alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    # A subprocess keeps Settings' cached DATABASE_URL out of this process.
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    admin = create_engine(_ADMIN_URL, isolation_level="AUTOCOMMIT")
    name = f"cag_test_{uuid.uuid4().hex[:12]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres is not reachable: {type(exc).__name__}")
    url = make_url(_ADMIN_URL).set(database=name).render_as_string(hide_password=False)
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def migrated_url(database_url: str) -> str:
    _alembic(database_url, "upgrade", "head")
    return database_url


@pytest.fixture
def session_factory(migrated_url: str) -> Iterator[sessionmaker]:
    engine = create_engine(migrated_url)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def test_migration_round_trip(database_url: str) -> None:
    engine = create_engine(database_url)
    try:
        _alembic(database_url, "upgrade", "head")
        expected_tables = {"pseudonym_mappings", "ingestion_jobs", "documents", "chunks"}
        assert expected_tables <= set(inspect(engine).get_table_names())
        assert any(
            index["name"] == "ix_documents_source_path" and index["unique"]
            for index in inspect(engine).get_indexes("documents")
        )
        with engine.connect() as connection:
            assert connection.scalar(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ) == 1

        _alembic(database_url, "downgrade", "base")
        assert not expected_tables & set(inspect(engine).get_table_names())

        _alembic(database_url, "upgrade", "head")
        assert expected_tables <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_async_document_ingestion_commits_or_rolls_back_as_a_unit(migrated_url: str) -> None:
    budget_path = os.path.join(os.path.dirname(__file__), "..", "data", "budgets_sample.json")
    with open(budget_path, encoding="utf-8") as budget_file:
        budget = Budget.model_validate(json.load(budget_file)[0])
    source_path = f"data/budgets/test-{uuid.uuid4().hex}.json"
    request = IngestRequest(
        source_path=source_path,
        document_type="historical_budget",
        content=budget,
    )
    failed_request = request.model_copy(update={"source_path": f"{source_path}.failed"})

    class FakeEmbedder:
        calls = 0
        query_calls = 0

        def embed_all(self, chunks):
            self.calls += 1
            return [
                EmbeddedChunk(**chunk.model_dump(), embedding=[1.0] * 1536)
                for chunk in chunks
            ]

        def embed_one(self, query):
            self.query_calls += 1
            return [1.0] * 1536

    class FailingEmbedder:
        def embed_all(self, chunks):
            raise RuntimeError("embedding failed")

    async def exercise() -> None:
        async_url = make_url(migrated_url).set(drivername="postgresql+asyncpg")
        engine = create_async_engine(async_url, pool_pre_ping=True)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        embedder = FakeEmbedder()
        chunker = JSONStructuralChunker()
        try:
            async with factory() as session:
                result = await ingest_document(request, session, chunker, embedder)
            assert result.document_id > 0
            assert result.chunks_created == len(budget.components)
            assert result.embedding_dimension == 1536
            assert embedder.calls == 1

            async with factory() as session:
                chunk_count = await session.scalar(
                    select(func.count()).select_from(ChunkRow).where(
                        ChunkRow.document_id == result.document_id
                    )
                )
            assert chunk_count == result.chunks_created

            async with factory() as session:
                search_response = await search_chunks(
                    SearchRequest(query="OAuth authentication", k=100),
                    session,
                    embedder,
                )
            assert search_response.query == "OAuth authentication"
            assert search_response.k == 100
            assert len(search_response.results) == result.chunks_created
            assert all(result.distance == pytest.approx(0.0) for result in search_response.results)
            assert embedder.query_calls == 1

            async with factory() as session:
                with pytest.raises(DuplicateDocumentError) as duplicate:
                    await ingest_document(request, session, chunker, embedder)
            assert duplicate.value.document_id == result.document_id
            assert embedder.calls == 1

            async with factory() as session:
                with pytest.raises(RuntimeError, match="embedding failed"):
                    await ingest_document(
                        failed_request, session, chunker, FailingEmbedder()
                    )
            async with factory() as session:
                failed_document = await session.scalar(
                    select(DocumentRow).where(
                        DocumentRow.source_path == failed_request.source_path
                    )
                )
            assert failed_document is None
        finally:
            await engine.dispose()

    asyncio.run(exercise())


def test_upgrade_is_idempotent(migrated_url: str) -> None:
    _alembic(migrated_url, "upgrade", "head")
    _alembic(migrated_url, "upgrade", "head")


def test_mapping_is_created_once_and_reused(session_factory) -> None:
    key = f"PERSON-{uuid.uuid4().hex}"
    with session_factory() as session:
        repo = MappingsRepository(session)

        first = repo.lookup_or_create("PERSON", key, lambda: "Marta Gil")
        second = repo.lookup_or_create("PERSON", key, lambda: "Other")

    assert first.pseudonym == second.pseudonym == "Marta Gil"


def test_the_factory_only_runs_on_a_miss(session_factory) -> None:
    key = f"EMAIL-{uuid.uuid4().hex}"
    calls: list[int] = []

    def factory() -> str:
        calls.append(1)
        return "a@b.c"

    with session_factory() as session:
        repo = MappingsRepository(session)
        repo.lookup_or_create("EMAIL_ADDRESS", key, factory)
        repo.lookup_or_create("EMAIL_ADDRESS", key, factory)

    assert len(calls) == 1


def test_same_hash_under_two_entity_types_is_independent(session_factory) -> None:
    key = uuid.uuid4().hex
    with session_factory() as session:
        repo = MappingsRepository(session)

        person = repo.lookup_or_create("PERSON", key, lambda: "P")
        place = repo.lookup_or_create("LOCATION", key, lambda: "L")

    assert (person.pseudonym, place.pseudonym) == ("P", "L")


def test_concurrent_writers_converge_to_one_pseudonym(session_factory) -> None:
    key = f"RACE-{uuid.uuid4().hex}"
    writers = 8
    barrier = threading.Barrier(writers)
    results: list[str] = []
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            with session_factory() as session:
                barrier.wait()
                mapping = MappingsRepository(session).lookup_or_create(
                    "PERSON", key, lambda: f"pseudonym-{index}"
                )
                results.append(mapping.pseudonym)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == writers
    assert len(set(results)) == 1


def test_forget_deletes_the_mapping(session_factory) -> None:
    key = f"FORGET-{uuid.uuid4().hex}"
    with session_factory() as session:
        repo = MappingsRepository(session)
        repo.lookup_or_create("PERSON", key, lambda: "X")

        assert repo.forget("PERSON", key) is True
        assert repo.lookup("PERSON", key) is None
        assert repo.forget("PERSON", key) is False


def test_postgres_mapping_store_satisfies_the_store_contract(session_factory) -> None:
    key = f"STORE-{uuid.uuid4().hex}"
    with session_factory() as session:
        store = PostgresMappingStore(session)

        first = store.lookup_or_create("PERSON", key, lambda: "A")
        again = store.lookup_or_create("PERSON", key, lambda: "B")
        assert store.forget("PERSON", key) is True
        reborn = store.lookup_or_create("PERSON", key, lambda: "C")

    assert first == again == "A"
    assert reborn == "C"


def test_job_lifecycle_uses_the_postgres_uuid_column(session_factory) -> None:
    with session_factory() as session:
        repo = JobsRepository(session)

        job = repo.create("presupuestos_json")
        repo.mark_running(job.job_id)
        repo.mark_completed(job.job_id, documents_count=6)

        done = repo.get(job.job_id)

    assert isinstance(done.job_id, uuid.UUID)
    assert (done.status, done.documents_count) == ("completed", 6)
    assert done.finished_at is not None
