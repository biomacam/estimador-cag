"""Smoke test of the stress runner and report against an in-process fake LLM (no network)."""

import copy
import csv
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services import llm_service, metadata_extractor
from app.services.llm_service import EstimationService
from app.sessions import ProjectMetadata, SessionStore
from evals.stress import run as stress_run
from evals.stress.fixtures.build_pdfs import build_all
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric
from evals.stress.report import load_rows, render
from evals.stress.scenarios import ScenarioTurn, get_scenario


def _fake_estimate(*, messages):
    result = EstimationResult(
        summary="Procurement portal estimation.",
        confidence_pct=80,
        phases=[{"name": "Build", "duration_weeks": 4, "cost_eur": 10_000, "summary": "Build the portal."}],
        total_duration_weeks=4,
        total_cost_eur=10_000,
    )
    chars = sum(len(m["content"]) for m in messages)
    return result, {"cache_hit": False, "tokens_in": chars // 4, "tokens_out": 50, "cost_usd": 0.001, "latency_ms": 10}


def _fake_extract(messages):
    transcript = messages[-1]["content"]
    metadata = ProjectMetadata(project_name="Nimbus") if "Nimbus" in transcript else ProjectMetadata()
    return metadata, {"latency_ms": 1}


@pytest.fixture
def runner_client(monkeypatch):
    get_settings.cache_clear()
    store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    monkeypatch.setattr(llm_service, "_invoke_structured_chat", _fake_estimate)
    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", _fake_extract)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def _run(client, scenario, size_kb, path: Path, repeat=0) -> list[dict]:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=stress_run.CSV_COLUMNS)
        writer.writeheader()
        stress_run._run_one_session(
            client, scenario, size_kb, repeat, LatencyBudgetMetric(60_000), CostBudgetMetric(10.0), writer
        )
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_runner_writes_one_row_per_turn_with_attachment_and_memory_columns(
    runner_client, tmp_path, monkeypatch
):
    build_all(tmp_path)
    monkeypatch.setattr(stress_run, "pdf_path", lambda kb: tmp_path / f"attach_{kb}kb.pdf")

    rows = _run(runner_client, get_scenario("growing"), 5, tmp_path / "out.csv")

    assert [int(r["turn_index"]) for r in rows] == list(range(1, 21))
    assert all(r["error"] == "" for r in rows)
    assert all(int(r["attachments_total_chars"]) > 5000 for r in rows)
    assert all(r["cache_hit_kind"] == "none" for r in rows)
    assert rows[0]["memory_drift_passed"] == "" and rows[0]["fact_recall"] == ""
    # The project name from turn 1 lives in metadata, so it is still found at turn 20.
    assert rows[19]["memory_drift_passed"] == "True"
    assert 0.0 < float(rows[19]["fact_recall"]) < 1.0
    assert int(rows[19]["messages_in_window"]) <= 12


def test_runner_records_error_row_and_stops_the_session(runner_client, tmp_path):
    pivot = get_scenario("pivot")
    scenario = replace(pivot, turns=[pivot.turns[0], ScenarioTurn(2, "too short", None), *pivot.turns[2:]])
    rows = _run(runner_client, scenario, 0, tmp_path / "out.csv")

    assert [r["error"] == "" for r in rows] == [True, False]
    assert "422" in rows[1]["error"]


def _bad_total_error():
    try:
        EstimationResult(
            summary="Procurement portal estimation.",
            confidence_pct=80,
            phases=[{"name": "Build", "duration_weeks": 4, "cost_eur": 49_140, "summary": "Build the portal."}],
            total_duration_weeks=4,
            total_cost_eur=60_000,
        )
    except ValueError as exc:
        return exc


def _first_two_turns(name):
    scenario = copy.copy(get_scenario(name))
    object.__setattr__(scenario, "turns", scenario.turns[:2])  # bypasses the 20-turn check
    return scenario


def test_a_502_is_retried_once_and_the_turn_is_recorded_with_its_retry(
    runner_client, tmp_path, monkeypatch
):
    calls = []

    def flaky(*, messages):
        calls.append(1)
        if len(calls) == 1:
            raise _bad_total_error()
        return _fake_estimate(messages=messages)

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", flaky)

    rows = _run(runner_client, _first_two_turns("growing"), 0, tmp_path / "out.csv")

    assert [r["error"] for r in rows] == ["", ""]
    assert [r["turn_retries"] for r in rows] == ["1", "0"]


def test_a_persistent_502_records_the_server_reason_and_stops_the_session(
    runner_client, tmp_path, monkeypatch
):
    def always_invalid(*, messages):
        raise _bad_total_error()

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", always_invalid)

    rows = _run(runner_client, _first_two_turns("growing"), 0, tmp_path / "out.csv")

    assert len(rows) == 1
    assert rows[0]["error"].startswith("HTTP 502 validation_failed")
    assert "phases sum" in rows[0]["error"]


def test_report_is_built_from_the_csv(runner_client, tmp_path):
    for repeat in range(2):
        rows = _run(runner_client, get_scenario("contradiction"), 0, tmp_path / f"r{repeat}.csv", repeat)
    parsed, errors = load_rows(tmp_path / "r1.csv")
    assert errors == 0 and len(parsed) == len(rows)

    report = render(parsed, errors)

    for heading in ("## 1. Summary table", "### 2a.", "### 2b.", "### 2c.", "## 3. Reading"):
        assert heading in report
    assert "| contradiction | 0 | 20 |" in report
