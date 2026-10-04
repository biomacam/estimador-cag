import pytest
from pydantic import ValidationError

from app.schemas.estimation import (
    EstimationResult,
    Phase,
    StructuredEstimationResponse,
)


def _valid_result(**overrides: object) -> dict[str, object]:
    payload = {
        "summary": "A small SaaS with authentication and an admin dashboard.",
        "confidence_pct": 70,
        "phases": [
            {
                "name": "Discovery",
                "duration_weeks": 1,
                "cost_eur": 5000,
                "summary": "Requirements workshops and technical scoping.",
            },
            {
                "name": "Implementation",
                "duration_weeks": 6,
                "cost_eur": 20000,
                "summary": "Build and integrate the core SaaS features.",
            },
        ],
        "total_duration_weeks": 7,
        "total_cost_eur": 25000,
    }
    payload.update(overrides)
    return payload


def test_valid_result_constructs_typed_phases_and_round_trips_json() -> None:
    result = EstimationResult.model_validate(_valid_result())

    assert all(isinstance(phase, Phase) for phase in result.phases)
    assert sum(phase.cost_eur for phase in result.phases) == result.total_cost_eur
    assert EstimationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("total_cost", [24999, 25001])
def test_phases_sum_must_equal_total_cost_exactly(total_cost: int) -> None:
    with pytest.raises(ValidationError, match="phases sum.*total_cost_eur"):
        EstimationResult.model_validate(_valid_result(total_cost_eur=total_cost))


@pytest.mark.parametrize("confidence", [0, 29])
def test_low_confidence_requires_out_of_scope_prefix(confidence: int) -> None:
    with pytest.raises(ValidationError, match="Out of scope:"):
        EstimationResult.model_validate(_valid_result(confidence_pct=confidence))


def test_low_confidence_with_correct_prefix_passes() -> None:
    result = EstimationResult.model_validate(
        _valid_result(
            confidence_pct=29,
            summary="Out of scope: the project description lacks essential requirements.",
        )
    )
    assert result.confidence_pct == 29


@pytest.mark.parametrize("confidence", [30, 100])
def test_confidence_at_or_above_threshold_accepts_regular_summary(confidence: int) -> None:
    result = EstimationResult.model_validate(_valid_result(confidence_pct=confidence))
    assert result.confidence_pct == confidence


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", "short"),
        ("confidence_pct", -1),
        ("confidence_pct", 101),
        ("phases", []),
        ("total_duration_weeks", 0),
        ("total_duration_weeks", 105),
        ("total_cost_eur", -1),
        ("total_cost_eur", 2000001),
    ],
)
def test_result_field_bounds_are_enforced(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult.model_validate(_valid_result(**{field: value}))
    assert any(error["loc"] == (field,) for error in exc_info.value.errors())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("duration_weeks", 0),
        ("duration_weeks", 53),
        ("cost_eur", -1),
        ("cost_eur", 1000001),
        ("summary", "short"),
    ],
)
def test_phase_field_bounds_are_enforced(field: str, value: object) -> None:
    phase = {
        "name": "Discovery",
        "duration_weeks": 1,
        "cost_eur": 5000,
        "summary": "Requirements workshops and technical scoping.",
    }
    with pytest.raises(ValidationError) as exc_info:
        Phase.model_validate({**phase, field: value})
    assert any(error["loc"] == (field,) for error in exc_info.value.errors())


def test_result_rejects_more_than_eight_phases() -> None:
    phase = {
        "name": "Discovery",
        "duration_weeks": 1,
        "cost_eur": 0,
        "summary": "Requirements workshops and technical scoping.",
    }
    with pytest.raises(ValidationError) as exc_info:
        EstimationResult.model_validate(
            _valid_result(phases=[phase] * 9, total_cost_eur=0)
        )
    assert any(error["loc"] == ("phases",) for error in exc_info.value.errors())


def test_structured_response_serializes_validated_result() -> None:
    response = StructuredEstimationResponse.model_validate(
        {"result": _valid_result(), "prompt_version": "v2"}
    )
    assert response.model_dump(mode="json") == {
        "result": _valid_result(),
        "prompt_version": "v2",
        "cached": False,
    }


def test_structured_response_rejects_inconsistent_nested_result() -> None:
    with pytest.raises(ValidationError, match="phases sum"):
        StructuredEstimationResponse.model_validate(
            {"result": _valid_result(total_cost_eur=25001), "prompt_version": "v1"}
        )