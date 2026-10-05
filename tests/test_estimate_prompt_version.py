"""Endpoint-level tests for the ?prompt_version= query param on POST /api/v1/estimate."""

import pytest
from fastapi.testclient import TestClient

from app.services import llm_service
from tests.helpers import valid_estimation_result

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

    def fake(*, system_prompt: str, user_message: str, **kwargs) -> tuple:
        calls.append({"system_prompt": system_prompt, "user_message": user_message})
        return valid_estimation_result(), {"cache_hit": False}

    monkeypatch.setattr(llm_service, "_invoke_structured_llm", fake)
    return calls


def test_default_prompt_version_is_v1(client: TestClient, stub_invoke_llm: list[dict]) -> None:
    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v1"


def test_query_param_selects_prompt_version_v2(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v2", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v2"
    # v2's distinctive tone must actually reach the system prompt sent to the LLM.
    assert "pragmatic engineering lead" in stub_invoke_llm[0]["system_prompt"]


def test_unknown_prompt_version_returns_422(
    client: TestClient, stub_invoke_llm: list[dict]
) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v99", json=VALID_PAYLOAD)

    assert response.status_code == 422
    assert stub_invoke_llm == []
