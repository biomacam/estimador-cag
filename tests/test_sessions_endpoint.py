from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions import ProjectMetadata, SessionStore


@pytest.fixture
def store() -> Iterator[SessionStore]:
    session_store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: session_store
    yield session_store
    app.dependency_overrides.pop(get_session_store, None)


def test_create_session_returns_a_uuid4_session_id(client: TestClient, store: SessionStore) -> None:
    response = client.post("/sessions")

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"session_id"}
    assert UUID(body["session_id"]).version == 4


def test_created_session_is_registered_in_the_store(client: TestClient, store: SessionStore) -> None:
    session_id = client.post("/sessions").json()["session_id"]

    session = store.get(session_id)

    assert session.history.messages == []
    assert session.metadata.project_name is None


def test_each_call_creates_a_new_session(client: TestClient, store: SessionStore) -> None:
    first = client.post("/sessions").json()["session_id"]
    second = client.post("/sessions").json()["session_id"]

    assert first != second
    assert len(store) == 2


def test_get_session_returns_metadata_apart_from_the_history_size(
    client: TestClient, store: SessionStore
) -> None:
    session = store.create()
    session.metadata = ProjectMetadata(project_name="Nimbus", mentioned_technologies=["React"])
    session.history.add("user", "hi")
    session.history.add("assistant", "hello")

    response = client.get(f"/sessions/{session.session_id}")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": session.session_id,
        "message_count": 2,
        "max_turns": session.history.max_turns,
        "metadata": {
            "project_name": "Nimbus",
            "assumed_team_size": None,
            "mentioned_technologies": ["React"],
            "agreed_scope": None,
        },
    }


def test_get_unknown_session_is_404(client: TestClient, store: SessionStore) -> None:
    response = client.get("/sessions/missing")

    assert response.status_code == 404
