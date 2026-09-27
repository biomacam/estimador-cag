from collections.abc import Iterator
import inspect

import pytest
from fastapi.testclient import TestClient
from litellm.exceptions import APIError

from app.context.examples import CANONICAL_EXAMPLES
from app.routers.estimations import create_estimation
from app.services import llm_service
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE
from app.services.llm_wrapper import LLMConfigurationError
from tests.helpers import unwrap_untrusted_input

TOTAL_COST = CANONICAL_EXAMPLES[0].total_cost

WELL_FORMED_MD = CANONICAL_EXAMPLES[0].estimation_markdown


def _fake_response(*, estimation: str = WELL_FORMED_MD, finish_reason: str = "stop") -> dict:
    return {
        "estimation": estimation,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": finish_reason,
        "usage": {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801},
        "latency_ms": 12,
        "cost_usd": 0.001234,
        "cache_hit": False,
    }


@pytest.fixture
def call_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict]]:
    """Replace the LLM seam with a recording fake. Returns the list of calls."""
    calls: list[dict] = []

    def fake(
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> dict:
        calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model_override": model_override,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            }
        )
        finish_reason = "length" if max_tokens <= 200 else "stop"
        return _fake_response(finish_reason=finish_reason)

    monkeypatch.setattr(llm_service, "_invoke_llm", fake)
    yield calls


def test_default_request_returns_validation(client: TestClient, call_log: list[dict]) -> None:
    payload = {"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["preprocessing"] == "none"
    assert body["finish_reason"] == "stop"
    # Data/instructions separation: the CAG examples ride in the system prompt,
    # the raw transcription reaches the LLM inside the untrusted-data boundary.
    assert CANONICAL_EXAMPLES[0].estimation_markdown in call_log[0]["system_prompt"]
    assert unwrap_untrusted_input(call_log[0]["user_message"]) == payload["transcription"]
    assert body["validation"] is not None
    assert body["validation"]["score"] == 1.0
    assert body["extracted_requirements"] is None
    assert body["cache_hit"] is False
    assert body["cost_usd"] == pytest.approx(0.001234)
    assert body["annual_maintenance"] == pytest.approx(TOTAL_COST * 0.12)
    assert "Annual maintenance" in body["estimation"]
    assert len(call_log) == 1


def test_two_phase_invokes_llm_twice_and_fills_extracted(
    client: TestClient, call_log: list[dict]
) -> None:
    payload = {
        "transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks.",
        "preprocessing": "two_phase",
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["preprocessing"] == "two_phase"
    assert body["extracted_requirements"] is not None
    assert len(call_log) == 2
    # The second call's user message should be the extracted requirements,
    # not the original transcription.
    assert unwrap_untrusted_input(call_log[1]["user_message"]) == body["extracted_requirements"]


def test_max_tokens_low_rejects_truncated_response_instead_of_200(
    client: TestClient, call_log: list[dict]
) -> None:
    payload = {
        "transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks.",
        "max_tokens": 200,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 502
    assert "failed to generate the estimation" in response.text.lower()
    assert len(call_log) == 1


def test_transcription_over_50000_tokens_is_rejected_before_llm(
    client: TestClient, call_log: list[dict]
) -> None:
    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "hello " * 50_001},
    )

    assert response.status_code == 422
    assert "50,000 tokens" in response.text
    assert call_log == []


def test_example_format_json_returns_200(client: TestClient, call_log: list[dict]) -> None:
    payload = {
        "transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks.",
        "example_format": "json",
        "num_examples": 2,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["input_tokens"] > 0
    assert "Reference examples (JSON):" in call_log[0]["system_prompt"]


def test_model_override_is_passed_to_provider(
    client: TestClient, call_log: list[dict]
) -> None:
    payload = {
        "transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks.",
        "model": "gpt-4o",
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    assert call_log[0]["model_override"] == "gpt-4o"


def test_use_examples_false_omits_examples_block(
    client: TestClient, call_log: list[dict]
) -> None:
    payload = {
        "transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks.",
        "use_examples": False,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    system_prompt = call_log[0]["system_prompt"]
    assert "EXAMPLE 1" not in system_prompt
    assert "Reference examples" not in system_prompt


def test_provider_error_does_not_leak_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_fragment = "sk-proj-super-secret-fragment"

    def boom(**kwargs) -> dict:
        raise APIError(
            status_code=401, message=secret_fragment, llm_provider="openai", model="gpt-4o-mini"
        )

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    payload = {"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}
    response = client.post("/api/v1/estimate", json=payload)

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

    payload = {"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.text


def test_create_estimation_is_sync_so_fastapi_runs_it_in_a_threadpool() -> None:
    """Regression guard for the blocking-handler bug: generate_estimation() blocks
    on the LLM call, so this route must stay a plain ``def`` — an ``async def``
    here would run that blocking call directly on the event loop and freeze every
    other request on the worker, including /health, for its whole duration.
    """
    assert not inspect.iscoroutinefunction(create_estimation)
