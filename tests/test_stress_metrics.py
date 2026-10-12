import pytest

from app.schemas.estimation import TurnObservation
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


def _obs(**overrides) -> TurnObservation:
    base = dict(
        turn_index=1, session_id="s", enriched_transcript_chars=200, attachments_total_chars=0,
        messages_in_window=2, anchors_count=0, summary_chars=0, tokens_in=500, tokens_out=300,
        cost_usd=0.001, latency_ms=1500,
    )
    return TurnObservation(**{**base, **overrides})


def _snapshot(*, project_name=None, technologies=(), scope=None, summary=None, **extra) -> dict:
    return {
        "session_id": "s",
        "metadata": {
            "project_name": project_name,
            "mentioned_technologies": list(technologies),
            "agreed_scope": scope,
        },
        "summary": summary,
        **extra,
    }


def test_latency_under_budget_passes():
    result = LatencyBudgetMetric(budget_ms=8000).evaluate(_obs(latency_ms=3200))
    assert (result.passed, result.score) == (True, 1.0)
    assert "3200" in result.details


def test_latency_over_budget_fails():
    result = LatencyBudgetMetric(budget_ms=8000).evaluate(_obs(latency_ms=9100))
    assert (result.passed, result.score) == (False, 0.0)


def test_latency_exactly_at_budget_passes():
    assert LatencyBudgetMetric(budget_ms=4000).evaluate(_obs(latency_ms=4000)).passed


def test_cost_under_budget_passes():
    assert CostBudgetMetric(budget_usd=0.02).evaluate(_obs(cost_usd=0.0035)).passed


def test_cost_over_budget_fails():
    result = CostBudgetMetric(budget_usd=0.002).evaluate(_obs(cost_usd=0.01))
    assert (result.passed, result.score) == (False, 0.0)


def test_cost_exactly_at_budget_passes():
    assert CostBudgetMetric(budget_usd=0.005).evaluate(_obs(cost_usd=0.005)).passed


@pytest.mark.parametrize("metric", [LatencyBudgetMetric, CostBudgetMetric])
def test_budgets_must_be_positive(metric):
    with pytest.raises(ValueError):
        metric(0)


def test_drift_fact_in_project_name_passes_case_insensitively():
    metric = MemoryDriftMetric("NIMBUS", "project_name")
    assert metric.evaluate(_snapshot(project_name="nimbus")).passed


def test_drift_fact_missing_fails():
    assert not MemoryDriftMetric("Nimbus", "project_name").evaluate(_snapshot(project_name="Atlas")).passed


def test_drift_technologies_accumulating_both_stacks_still_finds_the_new_one():
    snapshot = _snapshot(technologies=["React Native", "Flutter"])
    assert MemoryDriftMetric("Flutter", "technologies").evaluate(snapshot).passed
    assert MemoryDriftMetric("React Native", "technologies").evaluate(snapshot).passed


def test_drift_summary_scope_only_searches_the_summary():
    snapshot = _snapshot(project_name="Budget 80000", summary="Client confirmed 80000 EUR.")
    assert MemoryDriftMetric("80000", "summary").evaluate(snapshot).passed
    assert not MemoryDriftMetric("80000", "scope").evaluate(snapshot).passed


def test_drift_any_searches_anchors_too():
    snapshot = _snapshot(anchors=[{"role": "user", "content": "We signed an NDA last week."}])
    assert MemoryDriftMetric("signed an NDA").evaluate(snapshot).passed
    assert not MemoryDriftMetric("Cerner").evaluate(snapshot).passed


def test_drift_handles_empty_metadata_and_rejects_empty_fact():
    assert not MemoryDriftMetric("x", "technologies").evaluate({"metadata": None}).passed
    with pytest.raises(ValueError):
        MemoryDriftMetric("")
