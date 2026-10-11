"""Endpoint tests for POST /sessions/{session_id}/estimate-acb (Actor-Critic-Boss)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.critic import CriticFeedback, CriticIssue
from app.services import llm_service, metadata_extractor
from app.services.llm_service import EstimationService
from app.sessions import ProjectMetadata, SessionStore
from tests.helpers import valid_estimation_result

TRANSCRIPT = "We need a customer portal with invoices and a reporting dashboard."


@pytest.fixture
def store() -> Iterator[SessionStore]:
    session_store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    yield session_store
    app.dependency_overrides.pop(get_session_store, None)
    app.dependency_overrides.pop(get_estimation_service, None)


@pytest.fixture
def actor_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, str]]]:
    calls: list[list[dict[str, str]]] = []

    def fake(*, messages: list[dict[str, str]]) -> tuple:
        calls.append(messages)
        return valid_estimation_result(), {"cache_hit": False}

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", fake)
    monkeypatch.setattr(
        metadata_extractor, "_invoke_extractor", lambda messages: (ProjectMetadata(), {})
    )
    return calls


@pytest.fixture
def critic(monkeypatch: pytest.MonkeyPatch) -> list[CriticFeedback]:
    """Queue of critic reviews; the last one repeats once the queue has a single item."""
    reviews: list[CriticFeedback] = []

    class Llm:
        def complete_structured_chat(self, **kwargs):
            return (reviews.pop(0) if len(reviews) > 1 else reviews[0]), {}

    monkeypatch.setattr(llm_service, "_critic_llm", lambda: Llm())
    return reviews


def _math_error() -> CriticFeedback:
    return CriticFeedback(
        verdict="needs_iteration",
        issues=[CriticIssue(
            category="math_error", severity="critical", field_path="total_cost_eur",
            description="sum mismatch in the totals", suggested_fix="recompute the sum",
        )],
        confidence_in_review=80,
    )


def _post(client: TestClient, session_id: str):
    return client.post(f"/sessions/{session_id}/estimate-acb", data={"transcript": TRANSCRIPT})


def test_accept_on_first_pass_returns_the_draft_and_the_trace(
    client: TestClient, store: SessionStore, actor_calls: list, critic: list
) -> None:
    critic.append(CriticFeedback(verdict="accept", issues=[], confidence_in_review=90))
    session_id = store.create().session_id

    response = _post(client, session_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == valid_estimation_result().model_dump(mode="json")
    assert body["cached"] is False
    assert body["acb"]["final_decision"] == "accept"
    assert body["acb"]["iterations_run"] == 1
    assert body["acb"]["iterations"][0]["critic_verdict"] == "accept"
    assert len(actor_calls) == 1


def test_retry_sends_the_critic_feedback_to_the_actor(
    client: TestClient, store: SessionStore, actor_calls: list, critic: list
) -> None:
    critic.extend([_math_error(), CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)])

    body = _post(client, store.create().session_id).json()

    assert body["acb"]["final_decision"] == "accept"
    assert body["acb"]["iterations_run"] == 2
    assert "<critic_feedback>" not in actor_calls[0][-1]["content"]
    assert "math_error at `total_cost_eur`" in actor_calls[1][-1]["content"]


def test_persistent_issues_synthesize_the_last_draft_with_caveats(
    client: TestClient, store: SessionStore, actor_calls: list, critic: list
) -> None:
    critic.append(_math_error())

    body = _post(client, store.create().session_id).json()

    assert body["acb"]["final_decision"] == "synthesize"
    assert body["acb"]["iterations_run"] == 3
    assert body["result"]["summary"].startswith("⚠ Open caveats")
    assert "math_error" in body["result"]["summary"]
    assert body["result"]["total_cost_eur"] == valid_estimation_result().total_cost_eur
    assert body["result"]["confidence_pct"] >= 30
    assert body["text"].startswith("## Project summary\n⚠ Open caveats")


def test_only_the_final_result_is_persisted_without_critic_feedback(
    client: TestClient, store: SessionStore, actor_calls: list, critic: list
) -> None:
    critic.extend([_math_error(), CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)])
    session_id = store.create().session_id

    _post(client, session_id)

    history = store.get(session_id).history
    assert [m.role for m in history.messages] == ["user", "assistant"]
    assert "<critic_feedback>" not in history.messages[0].content
    assert store.get(session_id).last_resolved_tier == "default"


def test_critic_outage_accepts_the_first_draft(
    client: TestClient, store: SessionStore, actor_calls: list
) -> None:
    body = _post(client, store.create().session_id).json()

    assert body["acb"]["final_decision"] == "accept"
    assert body["acb"]["iterations"][0]["critic_confidence"] == 0


def test_unknown_session_is_404_and_does_not_call_the_llm(
    client: TestClient, store: SessionStore, actor_calls: list
) -> None:
    response = _post(client, "missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "session_not_found"
    assert actor_calls == []


def test_failed_actor_is_502_and_leaves_the_session_untouched(
    client: TestClient, store: SessionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*, messages):
        raise llm_service.LLMServiceError("boom")

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", failing)
    session_id = store.create().session_id

    response = _post(client, session_id)

    assert response.status_code == 502
    assert store.get(session_id).history.messages == []
