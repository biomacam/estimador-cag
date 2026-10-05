"""Tests for the output guardrail (filter policy, never raises).

``enforce_scope_response`` only matters for a combination the schema
validators normally forbid at construction time (low confidence without the
"Out of scope:" prefix), so these fixtures use ``model_construct`` to bypass
validation and simulate that edge case directly.
"""

from __future__ import annotations

import pytest

from app.guardrails.output import enforce_scope_response
from app.schemas.estimation import OUT_OF_SCOPE_PREFIX, EstimationResult


def _build(*, confidence_pct: int, summary: str) -> EstimationResult:
    return EstimationResult.model_construct(
        summary=summary,
        confidence_pct=confidence_pct,
        phases=[
            {
                "name": "Discovery",
                "duration_weeks": 1,
                "cost_eur": 2500,
                "summary": "Workshop and scoping.",
            }
        ],
        total_duration_weeks=1,
        total_cost_eur=2500,
    )


def test_high_confidence_result_passes_through_unchanged() -> None:
    original = _build(confidence_pct=70, summary="A small SaaS with auth and a dashboard.")
    assert enforce_scope_response(original) is original


def test_low_confidence_already_marked_passes_through_unchanged() -> None:
    original = _build(
        confidence_pct=10,
        summary=f"{OUT_OF_SCOPE_PREFIX} the description lacks essential requirements.",
    )
    assert enforce_scope_response(original) is original


def test_low_confidence_without_prefix_is_rewritten_into_a_placeholder() -> None:
    original = _build(confidence_pct=10, summary="A vague idea for some kind of app.")
    filtered = enforce_scope_response(original)

    assert filtered is not original
    assert filtered.summary.startswith(OUT_OF_SCOPE_PREFIX)
    assert "A vague idea for some kind of app." in filtered.summary
    assert filtered.confidence_pct == 10
    assert filtered.total_cost_eur == 0
    assert filtered.total_duration_weeks == 1
    assert len(filtered.phases) == 1
    assert filtered.phases[0].name == "Not estimated"
    assert filtered.phases[0].cost_eur == 0


def test_filtered_summary_is_truncated_to_the_schema_limit() -> None:
    original = _build(confidence_pct=0, summary="x" * 50 + " reason " * 200)
    filtered = enforce_scope_response(original)
    assert len(filtered.summary) <= 1200


def test_filter_never_raises_even_with_pathological_input() -> None:
    """No matter the field combination, the rewrite is a valid EstimationResult."""
    original = _build(confidence_pct=0, summary="x" * 1000)
    filtered = enforce_scope_response(original)
    EstimationResult.model_validate(filtered.model_dump())


@pytest.mark.parametrize("confidence", [29, 0])
def test_any_confidence_below_threshold_triggers_the_filter(confidence: int) -> None:
    original = _build(confidence_pct=confidence, summary="Not enough detail to size this.")
    filtered = enforce_scope_response(original)
    assert filtered.summary.startswith(OUT_OF_SCOPE_PREFIX)


@pytest.mark.parametrize("confidence", [30, 100])
def test_confidence_at_or_above_threshold_is_never_filtered(confidence: int) -> None:
    original = _build(confidence_pct=confidence, summary="Solid mid-size SaaS build.")
    assert enforce_scope_response(original) is original
