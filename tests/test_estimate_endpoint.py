"""Endpoint-level tests for POST /api/v1/estimate (Session 4 typed contract)."""

import inspect

import pytest
from fastapi.testclient import TestClient
from litellm.exceptions import APIError

from app.routers.estimations import create_estimation
from app.services import llm_service
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE
from app.services.llm_wrapper import LLMConfigurationError, LLMTruncatedResponseError
from tests.helpers import unwrap_untrusted_input

VALID_PAYLOAD = {
    "description": "Build a booking system for a yoga studio with class scheduling.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "line_items",
}


@pytest.fixture
def stub_invoke_llm(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the LLM seam with a recording fake so no provider is called."""
    calls: list[dict] = []

    def fake(*, system_prompt: str, user_message: str, **kwargs) -> dict:
        calls.append({"system_prompt": system_prompt, "user_message": user_message})
        return {
            "estimation": "## Fake estimation\n...",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
            "latency_ms": 5,
            "cost_usd": 0.0001,
            "cache_hit": False,
        }

    monkeypatch.setattr(llm_service, "_invoke_llm", fake)
    return calls


def test_valid_request_returns_the_generated_text(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"text": "## Fake estimation\n...", "prompt_version": "v1"}
    assert len(stub_invoke_llm) == 1
    # description reaches the LLM inside the untrusted-data boundary, not raw.
    assert unwrap_untrusted_input(stub_invoke_llm[0]["user_message"]) == VALID_PAYLOAD["description"]


def test_description_shorter_than_20_chars_is_rejected_before_llm(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    response = client.post("/api/v1/estimate", json={**VALID_PAYLOAD, "description": "too short"})

    assert response.status_code == 422
    assert stub_invoke_llm == []


def test_missing_required_field_returns_422(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "project_type"}
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 422
    assert stub_invoke_llm == []


def test_invalid_enum_value_returns_422(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    response = client.post("/api/v1/estimate", json={**VALID_PAYLOAD, "output_format": "pdf"})

    assert response.status_code == 422
    assert stub_invoke_llm == []


def test_truncated_response_is_rejected_instead_of_returned_as_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def truncated(**kwargs) -> dict:
        raise LLMTruncatedResponseError("Provider stopped before completing the response")

    monkeypatch.setattr(llm_service, "_invoke_llm", truncated)

    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 502
    assert LLM_PROVIDER_ERROR_MESSAGE in response.text


def test_provider_error_does_not_leak_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_fragment = "sk-proj-super-secret-fragment"

    def boom(**kwargs) -> dict:
        raise APIError(
            status_code=401, message=secret_fragment, llm_provider="openai", model="gpt-4o-mini"
        )

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 502
    assert secret_fragment not in response.text
    assert LLM_PROVIDER_ERROR_MESSAGE in response.text


def test_missing_llm_configuration_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_not_configured() -> None:
        raise LLMConfigurationError(
            "No LLM provider is configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    monkeypatch.setattr(llm_service, "get_llm_wrapper", raise_not_configured)

    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.text


def test_create_estimation_is_sync_so_fastapi_runs_it_in_a_threadpool() -> None:
    """Regression guard: generate_typed_estimation() blocks on the LLM call, so this
    route must stay a plain ``def`` — an ``async def`` here would run that blocking
    call directly on the event loop and freeze every other request on the worker,
    including /health, for its whole duration.
    """
    assert not inspect.iscoroutinefunction(create_estimation)
