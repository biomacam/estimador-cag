"""HTTP-level tests for the ingestion endpoints.

Postgres is replaced by an in-memory SQLite database shared by the request session
and the BackgroundTask session, so the router, orchestrator and ``JobsRepository``
run for real. TestClient executes background tasks before returning the response.
"""

from __future__ import annotations

import json
import textwrap
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.dependencies import get_catalog, get_filesystem_loader, get_parser_registry
from app.ingestion.catalog.loader import load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import default_registry
from app.main import app
from app.persistence.database import get_session
from app.persistence.models import Base

_CATALOG = """
version: "1.0.0"
sources:
  - name: tiny_json
    location: budgets
    format: json
    quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: include
  - name: broken_json
    location: not_there
    format: json
    quality: {completeness: 4, consistency: 4, actuality: 5, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: include
  - name: excluded_xlsx
    location: rate_card.xlsx
    format: xlsx
    quality: {completeness: 4, consistency: 5, actuality: 1, reliability: 4}
    sensitivity: {has_pii: false, pii_flags: []}
    decision: exclude
    decision_reason: "actuality=1, outdated"
"""


@pytest.fixture
def ingestion_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(textwrap.dedent(_CATALOG), encoding="utf-8")
    (tmp_path / "budgets").mkdir()
    for n in (1, 2):
        payload = {"budget_id": f"BUDGET-2024-000{n}", "currency": "EUR", "total_amount": n}
        (tmp_path / "budgets" / f"b{n}.json").write_text(json.dumps(payload), encoding="utf-8")

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def _request_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    catalog = load_catalog(catalog_path)
    loader = FileSystemLoader(data_root=tmp_path)
    registry = default_registry()
    app.dependency_overrides[get_catalog] = lambda: catalog
    app.dependency_overrides[get_filesystem_loader] = lambda: loader
    app.dependency_overrides[get_parser_registry] = lambda: registry
    app.dependency_overrides[get_session] = _request_session
    monkeypatch.setattr("app.routers.ingestion.new_session", factory)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _run(client: TestClient, source_name: str):
    return client.post("/api/v1/ingestion/runs", json={"source_name": source_name})


def test_post_run_returns_202_and_creates_job(ingestion_client) -> None:
    response = _run(ingestion_client, "tiny_json")

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["source_name"] == "tiny_json"
    uuid.UUID(body["job_id"])


def test_get_job_reflects_completed_state(ingestion_client) -> None:
    job_id = _run(ingestion_client, "tiny_json").json()["job_id"]

    response = ingestion_client.get(f"/api/v1/ingestion/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["documents_count"] == 2
    assert body["error_message"] is None
    assert body["finished_at"] is not None


def test_failed_job_exposes_only_the_exception_type(ingestion_client, tmp_path) -> None:
    job_id = _run(ingestion_client, "broken_json").json()["job_id"]

    body = ingestion_client.get(f"/api/v1/ingestion/jobs/{job_id}").json()

    assert body["status"] == "failed"
    assert body["error_message"] == "FileNotFoundError"
    assert str(tmp_path) not in json.dumps(body)


def test_unknown_source_returns_404(ingestion_client) -> None:
    response = _run(ingestion_client, "does_not_exist")

    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "unknown_source"


def test_excluded_source_returns_400_with_the_reason(ingestion_client) -> None:
    response = _run(ingestion_client, "excluded_xlsx")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["reason"] == "source_not_included"
    assert detail["decision"] == "exclude"
    assert "outdated" in detail["decision_reason"]


def test_get_unknown_job_returns_404(ingestion_client) -> None:
    response = ingestion_client.get(f"/api/v1/ingestion/jobs/{uuid.uuid4()}")

    assert response.status_code == 404


def test_get_job_with_a_malformed_id_returns_422(ingestion_client) -> None:
    assert ingestion_client.get("/api/v1/ingestion/jobs/not-a-uuid").status_code == 422


@pytest.mark.parametrize(
    "body",
    [{}, {"source_name": ""}, {"source_name": "x" * 129}, {"source_name": "tiny_json", "extra": 1}],
)
def test_invalid_request_bodies_return_422(ingestion_client, body) -> None:
    assert ingestion_client.post("/api/v1/ingestion/runs", json=body).status_code == 422


def test_ingestion_routes_are_registered_in_the_openapi_schema(ingestion_client) -> None:
    paths = ingestion_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/ingestion/runs" in paths
    assert "/api/v1/ingestion/jobs/{job_id}" in paths
