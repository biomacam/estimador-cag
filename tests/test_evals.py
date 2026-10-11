"""Tests for the ``evals`` harness: dataset, deterministic metrics and the runner (no real LLM)."""

import json
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services import llm_service, metadata_extractor
from app.services.llm_service import EstimationService
from app.sessions import ProjectMetadata, SessionStore
from evals import run as evals_run
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import run_all_metrics


def _estimation(total: int = 30_000, weeks: int = 8, summary: str = "CRM build with React.") -> EstimationResult:
    return EstimationResult(
        summary=summary,
        confidence_pct=75,
        phases=[
            {"name": "Discovery", "duration_weeks": 2, "cost_eur": 5_000, "summary": "Workshops and scoping."},
            {"name": "Build", "duration_weeks": weeks - 4, "cost_eur": total - 10_000, "summary": "Core build on Postgres."},
            {"name": "Launch", "duration_weeks": 2, "cost_eur": 5_000, "summary": "QA and go-live."},
        ],
        total_duration_weeks=weeks,
        total_cost_eur=total,
    )


def _case(**overrides) -> GoldenCase:
    base = {
        "id": "case-test",
        "transcript": "A CRM for the sales team with React and Postgres.",
        "project_type": "web_saas",
        "expected_in_summary": ["CRM"],
        "expected_technologies_any_of": ["React", "Postgres"],
        "expected_phase_count_range": [3, 6],
        "expected_cost_range_eur": [15_000, 80_000],
        "expected_duration_weeks_range": [4, 16],
    }
    return GoldenCase.model_validate(base | overrides)


def _by_name(case: GoldenCase, result: EstimationResult) -> dict[str, bool]:
    return {m.name: m.passed for m in run_all_metrics(case, result)}


def test_dataset_has_15_plus_valid_cases_with_unique_ids_and_out_of_scope_mix() -> None:
    cases = load_dataset()

    assert len(cases) >= 15
    assert len({c.id for c in cases}) == len(cases)
    assert any(c.expected_out_of_scope for c in cases)


def test_all_metrics_pass_for_a_result_inside_every_expectation() -> None:
    assert _by_name(_case(), _estimation()) == {
        "schema_adherence": True, "cost_bounds": True, "content_recall": True,
    }


def test_metrics_report_why_a_case_fails() -> None:
    results = run_all_metrics(
        _case(expected_phase_count_range=[4, 6], expected_in_summary=["invoices"]),
        _estimation(total=100_000, summary="Generic build."),
    )

    details = {m.name: m.details for m in results}
    assert "phase count 3 outside [4, 6]" in details["schema_adherence"]
    assert "cost 100000 EUR outside [15000, 80000]" in details["cost_bounds"]
    assert "missing in summary: ['invoices']" in details["content_recall"]
    assert all(not m.passed and m.score == 0.0 for m in results)


def test_out_of_scope_cases_expect_the_placeholder_envelope_and_skip_recall() -> None:
    case = _case(expected_out_of_scope=True, expected_phase_count_range=None)
    envelope = EstimationResult.model_validate(evals_run._guardrail_payload("pii", "blocked")["result"])

    assert all(_by_name(case, envelope).values())
    assert not _by_name(case, _estimation())["cost_bounds"]


def test_guardrail_rejection_counts_only_for_out_of_scope_cases() -> None:
    rejected = httpx.Response(400, json={"detail": {"reason": "prompt_injection", "message": "nope"}})

    assert evals_run._maybe_guardrail_payload(_case(expected_out_of_scope=True), rejected) is not None
    assert evals_run._maybe_guardrail_payload(_case(), rejected) is None
    other = httpx.Response(400, json={"detail": "bad"})
    assert evals_run._maybe_guardrail_payload(_case(expected_out_of_scope=True), other) is None


def test_explicit_tier_is_sent_and_auto_is_omitted() -> None:
    assert "tier" not in evals_run._form_body(_case())
    assert evals_run._form_body(_case(tier="pm"))["tier"] == "pm"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        llm_service, "_invoke_structured_chat",
        lambda *, messages: (_estimation(), {"cache_hit": False}),
    )
    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", lambda messages: (ProjectMetadata(), {}))
    store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_session_store, None)
    app.dependency_overrides.pop(get_estimation_service, None)


@pytest.mark.parametrize("mode", ["actor", "acb"])
def test_run_cases_scores_every_case_through_the_api(api: TestClient, mode: str) -> None:
    cases = [_case(id="case-a"), _case(id="case-b", expected_cost_range_eur=[1, 2])]

    report = evals_run.run_cases(api, cases, mode)

    assert (report["cases_total"], report["cases_passing"]) == (2, 1)
    assert [row["passed"] for row in report["rows"]] == [True, False]
    assert report["rows"][0]["acb_final_decision"] == ("accept" if mode == "acb" else None)


def test_run_cases_records_errors_without_aborting(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*, messages):
        raise llm_service.LLMServiceError("boom")

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", failing)

    report = evals_run.run_cases(api, [_case(id="case-a"), _case(id="case-b")], "actor")

    assert report["cases_passing"] == 0
    assert all("error" in row for row in report["rows"])


def test_main_writes_the_report_and_exit_code_reflects_failures(
    api: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evals_run, "load_dataset", lambda: [_case()])
    monkeypatch.setattr(evals_run, "_in_process_client", lambda: nullcontext(api))
    output = tmp_path / "report.json"

    assert evals_run.main(["--mode", "actor", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["cases_passing"] == 1
