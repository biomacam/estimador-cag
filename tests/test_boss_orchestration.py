"""Tests for the Boss orchestrator — accept / iterate / synthesize routes."""

import pytest

from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import EstimationResult
from app.services.boss import Boss


def _result(total: int = 25_000, confidence: int = 70) -> EstimationResult:
    return EstimationResult(
        summary="Mid-size build description.",
        confidence_pct=confidence,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Workshops and tech spike for the project."},
            {"name": "Build", "duration_weeks": 5, "cost_eur": total - 5_000,
             "summary": "Core build of the agreed scope."},
        ],
        total_duration_weeks=6,
        total_cost_eur=total,
    )


def _accept() -> CriticFeedback:
    return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)


def _needs_iteration(category: str = "math_error") -> CriticFeedback:
    return CriticFeedback(
        verdict="needs_iteration",
        issues=[CriticIssue(
            category=category, severity="critical", field_path="total_cost_eur",
            description="sum mismatch", suggested_fix="recompute totals",
        )],
        confidence_in_review=80,
    )


def test_accept_first_pass_stops_after_one_iteration() -> None:
    actor_calls: list[CriticFeedback | None] = []

    def actor(feedback):
        actor_calls.append(feedback)
        return _result()

    result, trace = Boss(max_iterations=2).run(actor=actor, critic=lambda _: _accept())

    assert result.total_cost_eur == 25_000
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 1
    assert actor_calls == [None]


def test_iterate_then_accept_passes_the_feedback_to_the_actor() -> None:
    actor_calls: list[CriticFeedback | None] = []
    reviews = [_needs_iteration(), _accept()]

    def actor(feedback):
        actor_calls.append(feedback)
        return _result(total=25_000 + len(actor_calls))

    result, trace = Boss(max_iterations=3).run(actor=actor, critic=lambda _: reviews.pop(0))

    assert trace.final_decision == "accept"
    assert trace.iterations_run == 2
    assert actor_calls[0] is None
    assert actor_calls[1].issues[0].field_path == "total_cost_eur"
    assert result.total_cost_eur == 25_002


def test_exhausted_budget_keeps_the_last_draft_with_caveats() -> None:
    result, trace = Boss(max_iterations=2).run(
        actor=lambda _: _result(), critic=lambda _: _needs_iteration()
    )

    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == 2
    assert result.total_cost_eur == 25_000
    assert [p.name for p in result.phases] == ["Discovery", "Build"]
    assert result.summary.startswith("⚠ Open caveats")
    assert "[critical] math_error" in result.summary
    assert result.confidence_pct == 35


def test_confidence_is_floored_at_the_low_confidence_threshold() -> None:
    result, _ = Boss(max_iterations=1).run(
        actor=lambda _: _result(confidence=40), critic=lambda _: _needs_iteration()
    )

    assert result.confidence_pct == 30
    assert not result.summary.startswith("Out of scope:")


def test_reject_skips_to_synthesize_immediately() -> None:
    actor_calls = 0

    def actor(_):
        nonlocal actor_calls
        actor_calls += 1
        return _result()

    reject = CriticFeedback(
        verdict="reject",
        issues=[CriticIssue(
            category="scope_mismatch", severity="critical", field_path="summary",
            description="user asked for mobile but got web",
        )],
        confidence_in_review=85,
    )
    result, trace = Boss(max_iterations=4).run(actor=actor, critic=lambda _: reject)

    assert actor_calls == 1
    assert trace.final_decision == "synthesize"
    assert "scope_mismatch" in result.summary
    assert result.total_cost_eur == 25_000


def test_trace_records_one_entry_per_iteration() -> None:
    reviews = [_needs_iteration(), _needs_iteration("phase_imbalance"), _accept()]

    _, trace = Boss(max_iterations=4).run(actor=lambda _: _result(), critic=lambda _: reviews.pop(0))

    assert [it.decision_after for it in trace.iterations] == ["iterate", "iterate", "accept"]
    assert "math_error" in trace.iterations[0].issue_summary[0]
    assert "phase_imbalance" in trace.iterations[1].issue_summary[0]


def test_invalid_max_iterations_is_rejected() -> None:
    with pytest.raises(ValueError):
        Boss(max_iterations=0)
