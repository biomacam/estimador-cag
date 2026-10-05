"""Integration tests of the conversational flow through ``httpx.AsyncClient`` (ASGI, no network).

The provider is replaced by fakes whose answers depend on what the prompt contains,
so each test shows how information travels: attachment text and earlier turns reach
the LLM, and the metadata extracted from one turn comes back in the next one.
"""

import asyncio
from collections.abc import Iterator

import httpx
import pytest

from app.config import get_settings
from app.dependencies import get_estimation_service, get_session_store
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services import llm_service, metadata_extractor
from app.services.llm_service import EstimationService
from app.sessions import MAX_TURNS, ProjectMetadata
from tests.helpers import make_pdf

PORTAL = "We need a customer portal with invoices and a reporting dashboard."


class FakeLlm:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def estimate(self, *, messages: list[dict[str, str]]) -> tuple[EstimationResult, dict]:
        """Adds an SSO phase only when the prompt mentions SAML."""
        self.calls.append(messages)
        phases = [
            {"name": "Implementation", "duration_weeks": 4, "cost_eur": 10_000,
             "summary": "Build the core portal features."},
        ]
        if "SAML" in messages[-1]["content"]:
            phases.append(
                {"name": "SSO integration", "duration_weeks": 2, "cost_eur": 4_000,
                 "summary": "Integrate SAML single sign-on."}
            )
        result = EstimationResult(
            summary="Customer portal estimation.",
            confidence_pct=80,
            phases=phases,
            total_duration_weeks=sum(phase["duration_weeks"] for phase in phases),
            total_cost_eur=sum(phase["cost_eur"] for phase in phases),
        )
        return result, {"cache_hit": False}

    def extract(self, messages: list[dict[str, str]]) -> tuple[ProjectMetadata, dict]:
        """Reads only the latest user transcript, like the real extractor is asked to."""
        text = messages[-1]["content"]
        transcript = text.split("<latest_user_transcript>")[1].split("</latest_user_transcript>")[0]
        facts = ProjectMetadata()
        if "Nimbus Portal" in transcript:
            facts.project_name = "Nimbus Portal"
        if "team of 4" in transcript:
            facts.assumed_team_size = 4
        facts.mentioned_technologies = [t for t in ("React", "Postgres") if t in transcript]
        return facts, {"latency_ms": 1}


@pytest.fixture(autouse=True)
def wiring() -> Iterator[None]:
    """Fresh settings and session store per test, and no moderation call."""
    get_settings.cache_clear()
    get_session_store.cache_clear()
    app.dependency_overrides[get_estimation_service] = lambda: EstimationService()
    yield
    app.dependency_overrides.pop(get_estimation_service, None)
    get_settings.cache_clear()
    get_session_store.cache_clear()


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLlm:
    fake = FakeLlm()
    monkeypatch.setattr(llm_service, "_invoke_structured_chat", fake.estimate)
    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", fake.extract)
    return fake


def _api() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _new_session(api: httpx.AsyncClient) -> str:
    response = await api.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


async def _turn(
    api: httpx.AsyncClient, session_id: str, transcript: str, files: list | None = None
) -> httpx.Response:
    response = await api.post(
        f"/sessions/{session_id}/estimate", data={"transcript": transcript}, files=files
    )
    assert response.status_code == 200, response.text
    return response


async def _metadata(api: httpx.AsyncClient, session_id: str) -> dict:
    return (await api.get(f"/sessions/{session_id}")).json()["metadata"]


def test_two_requests_in_one_session_update_the_project_metadata(llm: FakeLlm) -> None:
    async def scenario() -> tuple[dict, dict]:
        async with _api() as api:
            session_id = await _new_session(api)
            await _turn(api, session_id, "The project is called Nimbus Portal and the frontend uses React.")
            after_first = await _metadata(api, session_id)
            await _turn(api, session_id, "We are a team of 4 and the backend uses Postgres.")
            return after_first, await _metadata(api, session_id)

    after_first, after_second = asyncio.run(scenario())

    assert after_first == {
        "project_name": "Nimbus Portal",
        "assumed_team_size": None,
        "mentioned_technologies": ["React"],
        "agreed_scope": None,
    }
    assert after_second == {
        "project_name": "Nimbus Portal",
        "assumed_team_size": 4,
        "mentioned_technologies": ["React", "Postgres"],
        "agreed_scope": None,
    }
    first_system, second_system = llm.calls[0][0]["content"], llm.calls[1][0]["content"]
    assert "<project_metadata>\n</project_metadata>" in first_system
    assert "Nimbus Portal" in second_system
    assert "React" in second_system


def test_a_pdf_attachment_changes_the_estimation(llm: FakeLlm) -> None:
    pdf = make_pdf("The portal must support SSO with SAML for all employees")

    async def scenario() -> tuple[dict, dict]:
        async with _api() as api:
            plain = await _turn(api, await _new_session(api), PORTAL)
            attached = await _turn(
                api,
                await _new_session(api),
                PORTAL,
                files=[("attachments", ("requirements.pdf", pdf, "application/pdf"))],
            )
        return plain.json()["result"], attached.json()["result"]

    plain, attached = asyncio.run(scenario())

    assert [phase["name"] for phase in plain["phases"]] == ["Implementation"]
    assert [phase["name"] for phase in attached["phases"]] == ["Implementation", "SSO integration"]
    assert attached["total_cost_eur"] > plain["total_cost_eur"]
    assert "--- attachment: requirements.pdf ---" in llm.calls[1][-1]["content"]


@pytest.mark.parametrize("max_turns", [3, MAX_TURNS])
def test_eight_turns_never_send_more_history_than_the_configured_window(
    llm: FakeLlm, monkeypatch: pytest.MonkeyPatch, max_turns: int
) -> None:
    monkeypatch.setenv("MAX_CONVERSATION_TURNS", str(max_turns))
    get_settings.cache_clear()
    get_session_store.cache_clear()

    async def scenario() -> None:
        async with _api() as api:
            session_id = await _new_session(api)
            for turn in range(1, 9):
                await _turn(api, session_id, f"Turn {turn}: keep refining the customer portal scope.")

    asyncio.run(scenario())

    assert len(llm.calls) == 8
    for number, messages in enumerate(llm.calls, start=1):
        history = messages[1:-1]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert len(history) <= 2 * max_turns
        assert len(history) == 2 * min(number - 1, max_turns)
        assert [m["role"] for m in history] == ["user", "assistant"] * (len(history) // 2)

    last_call = " ".join(message["content"] for message in llm.calls[-1])
    first_kept = 8 - max_turns
    assert f"Turn {first_kept}:" in last_call
    assert f"Turn {first_kept - 1}:" not in last_call
