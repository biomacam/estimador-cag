"""The 502 body carries a safe, fixed ``reason`` instead of provider exception text."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from instructor.core import InstructorRetryException
from instructor.v2.core.errors import FailedAttempt
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import ValidationError

from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services import llm_service
from app.services.llm_errors import classify_llm_error
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE, EstimationService
from app.services.llm_wrapper import LLMTruncatedResponseError
from app.sessions import SessionStore
from tests.helpers import valid_estimation_result

SECRET = "sk-proj-super-secret-fragment"
PAYLOAD = {
    "description": "Build a booking system for a yoga studio with class scheduling.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "line_items",
}


def _validation_error(summary: str = "Customer portal estimation.") -> ValidationError:
    payload = valid_estimation_result().model_dump()
    payload["total_cost_eur"] = 1
    payload["summary"] = summary
    with pytest.raises(ValidationError) as info:
        EstimationResult.model_validate(payload)
    return info.value


def _provider_error(kind: type[Exception]) -> Exception:
    return kind(message=SECRET, llm_provider="openai", model="gpt-4o-mini")


@pytest.fixture(autouse=True)
def offline_service() -> Iterator[None]:
    store = SessionStore()
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    app.dependency_overrides[get_session_store] = lambda: store
    yield
    app.dependency_overrides.pop(get_estimation_service, None)
    app.dependency_overrides.pop(get_session_store, None)


def test_validation_reason_keeps_the_validator_message_without_the_model_output() -> None:
    reason = classify_llm_error(_validation_error(summary=SECRET))

    assert reason.code == "validation_failed"
    assert "phases sum (10000 EUR) does not match total_cost_eur (1 EUR)" in reason.message
    assert SECRET not in reason.message


def test_instructor_wrapper_is_unwrapped_through_its_failed_attempts() -> None:
    error = _validation_error()
    wrapper = InstructorRetryException(
        "Max retries exceeded",
        n_attempts=1,
        total_usage=None,
        failed_attempts=[FailedAttempt(1, error)],
    )

    assert classify_llm_error(wrapper).code == "validation_failed"


def test_instructor_wrapper_is_unwrapped_through_its_cause() -> None:
    try:
        raise InstructorRetryException(
            "Max retries exceeded", n_attempts=1, total_usage=None
        ) from _provider_error(AuthenticationError)
    except InstructorRetryException as wrapper:
        assert classify_llm_error(wrapper).code == "authentication_failed"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_provider_error(AuthenticationError), "authentication_failed"),
        (_provider_error(RateLimitError), "rate_limited"),
        (_provider_error(Timeout), "timeout"),
        (_provider_error(ContextWindowExceededError), "context_too_long"),
        (_provider_error(ServiceUnavailableError), "provider_unavailable"),
        (APIError(status_code=500, message=SECRET, llm_provider="openai", model="gpt"), "provider_error"),
    ],
)
def test_provider_errors_map_to_a_code_without_any_message(error: Exception, code: str) -> None:
    reason = classify_llm_error(error)

    assert reason.as_dict() == {"code": code}
    assert SECRET not in str(reason.as_dict())


@pytest.mark.parametrize(
    "error",
    [json.JSONDecodeError("bad", "{", 1), LLMTruncatedResponseError("cut")],
)
def test_incomplete_output_maps_to_incomplete_response(error: Exception) -> None:
    assert classify_llm_error(error).code == "incomplete_response"


def test_anything_else_is_an_unexpected_error() -> None:
    assert classify_llm_error(RuntimeError(SECRET)).as_dict() == {"code": "unexpected_error"}


def test_the_estimate_endpoint_reports_a_validation_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**kwargs):
        raise _validation_error(summary=SECRET)

    monkeypatch.setattr(llm_service, "_invoke_structured_llm", failing)

    response = client.post("/api/v1/estimate", json=PAYLOAD)

    assert response.status_code == 502
    body = response.json()
    assert body["detail"] == LLM_PROVIDER_ERROR_MESSAGE
    assert body["reason"]["code"] == "validation_failed"
    assert "phases sum" in body["reason"]["message"]
    assert SECRET not in response.text


def test_the_estimate_endpoint_reports_only_the_code_for_provider_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**kwargs):
        raise _provider_error(AuthenticationError)

    monkeypatch.setattr(llm_service, "_invoke_structured_llm", failing)

    response = client.post("/api/v1/estimate", json=PAYLOAD)

    assert response.status_code == 502
    assert response.json() == {
        "detail": LLM_PROVIDER_ERROR_MESSAGE,
        "reason": {"code": "authentication_failed"},
    }
    assert SECRET not in response.text


def test_the_estimate_endpoint_reports_unexpected_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(llm_service, "_invoke_structured_llm", failing)

    response = client.post("/api/v1/estimate", json=PAYLOAD)

    assert response.status_code == 502
    assert response.json()["reason"] == {"code": "unexpected_error"}
    assert SECRET not in response.text


def test_the_session_endpoint_reports_the_reason_too(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*, messages):
        raise _validation_error()

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", failing)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/estimate",
        data={"transcript": "We need a customer portal with invoices and reports."},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["detail"] == LLM_PROVIDER_ERROR_MESSAGE
    assert body["reason"]["code"] == "validation_failed"
    assert "phases sum (10000 EUR) does not match total_cost_eur (1 EUR)" in body["reason"]["message"]
