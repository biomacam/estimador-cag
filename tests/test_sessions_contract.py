"""Contract details of the session endpoints: scope replacement, response fields, 404 detail
and the number of LLM calls per turn."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.services import llm_service, metadata_extractor
from app.services.llm_service import EstimationService
from app.sessions import ProjectMetadata, SessionStore
from tests.helpers import valid_estimation_result

TRANSCRIPT = "We need a customer portal with invoices and a reporting dashboard."


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    store = SessionStore()
    estimations: list[list[dict[str, str]]] = []
    extractions: list[list[dict[str, str]]] = []
    scripted: list[ProjectMetadata] = []

    def estimate(*, messages: list[dict[str, str]]) -> tuple:
        estimations.append(messages)
        return valid_estimation_result(), {"cache_hit": False}

    def extract(messages: list[dict[str, str]]) -> tuple:
        extractions.append(messages)
        return (scripted.pop(0) if scripted else ProjectMetadata()), {"latency_ms": 1}

    monkeypatch.setattr(llm_service, "_invoke_structured_chat", estimate)
    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", extract)
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    yield SimpleNamespace(
        store=store, estimations=estimations, extractions=extractions, scripted=scripted
    )
    app.dependency_overrides.pop(get_session_store, None)
    app.dependency_overrides.pop(get_estimation_service, None)


def _new_session(client: TestClient) -> str:
    return client.post("/sessions").json()["session_id"]


def _turn(client: TestClient, session_id: str, query: str = ""):
    return client.post(f"/sessions/{session_id}/estimate{query}", data={"transcript": TRANSCRIPT})


def test_merge_replaces_agreed_scope_with_a_new_non_null_value() -> None:
    base = ProjectMetadata(agreed_scope="Phase 1 MVP", project_name="CRM")

    merged = base.merge_with(ProjectMetadata(agreed_scope="Phase 1 MVP with billing"))

    assert merged.agreed_scope == "Phase 1 MVP with billing"
    assert merged.project_name == "CRM"


def test_a_new_agreed_scope_replaces_the_previous_one_across_turns(
    client: TestClient, api: SimpleNamespace
) -> None:
    api.scripted.append(ProjectMetadata(project_name="Nimbus", agreed_scope="Phase 1 MVP"))
    api.scripted.append(ProjectMetadata(agreed_scope="Phase 1 MVP with billing"))
    session_id = _new_session(client)

    _turn(client, session_id)
    _turn(client, session_id)

    metadata = api.store.get(session_id).metadata
    assert metadata.agreed_scope == "Phase 1 MVP with billing"
    assert metadata.project_name == "Nimbus"


def test_the_response_reports_the_prompt_version_and_that_it_is_not_cached(
    client: TestClient, api: SimpleNamespace
) -> None:
    session_id = _new_session(client)

    default = _turn(client, session_id).json()
    v2 = _turn(client, session_id, "?prompt_version=v2").json()

    assert default["prompt_version"] == "v1"
    assert v2["prompt_version"] == "v2"
    assert default["cached"] is False
    assert v2["cached"] is False


def test_unknown_session_detail_is_session_not_found(
    client: TestClient, api: SimpleNamespace
) -> None:
    response = _turn(client, "missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "session_not_found"
    assert api.estimations == []


def test_each_turn_makes_one_estimation_call_and_one_metadata_extraction_call(
    client: TestClient, api: SimpleNamespace
) -> None:
    session_id = _new_session(client)

    for _ in range(3):
        assert _turn(client, session_id).status_code == 200

    assert len(api.estimations) == 3
    assert len(api.extractions) == 3
