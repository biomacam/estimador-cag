"""Streamlit client tests: the HTTP calls to the estimator are faked."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
BASE = "http://localhost:8000"
INFO = {
    "session_id": "s-1",
    "message_count": 2,
    "max_turns": 6,
    "metadata": {
        "project_name": "Nimbus Portal",
        "assumed_team_size": 4,
        "mentioned_technologies": ["React"],
        "agreed_scope": "Invoices",
    },
}
RESULT = {"result": {}, "text": "RESULT_TEXT", "prompt_version": "v1", "cached": False}
TRANSCRIPT = "We need a customer portal with invoices and reports."


def _response(method: str, url: str, status: int, body: object) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request(method, url))


class FakeApi:
    def __init__(self) -> None:
        self.session_ids = ["s-1", "s-2", "s-3"]
        self.estimate_response: tuple[int, object] = (200, RESULT)
        self.estimate_calls: list[dict] = []
        self.created = 0

    def post(self, url: str, **kwargs) -> httpx.Response:
        if url.endswith("/estimate"):
            self.estimate_calls.append({"url": url, **kwargs})
            status, body = self.estimate_response
            return _response("POST", url, status, body)
        self.created += 1
        return _response("POST", url, 201, {"session_id": self.session_ids.pop(0)})

    def get(self, url: str, **kwargs) -> httpx.Response:
        return _response("GET", url, 200, {**INFO, "session_id": url.rsplit("/", 1)[-1]})


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    fake = FakeApi()
    monkeypatch.setattr(httpx, "post", fake.post)
    monkeypatch.setattr(httpx, "get", fake.get)
    return fake


def _app() -> AppTest:
    app = AppTest.from_file(APP_FILE, default_timeout=15).run()
    assert not app.exception
    return app


def _submit(app: AppTest, transcript: str = TRANSCRIPT) -> AppTest:
    app.text_area(key=f"transcript_{app.session_state['form_version']}").set_value(transcript)
    app.button(key="estimate").click().run()
    assert not app.exception
    return app


def test_a_session_is_created_on_load_and_kept_in_session_state(api: FakeApi) -> None:
    app = _app()

    assert app.session_state["session_id"] == "s-1"
    assert api.created == 1


def test_the_session_is_not_recreated_on_later_reruns(api: FakeApi) -> None:
    app = _app()

    app.run()

    assert api.created == 1
    assert app.session_state["session_id"] == "s-1"


def test_the_sidebar_shows_the_project_metadata_apart_from_the_history(api: FakeApi) -> None:
    app = _app()

    sidebar_json = " ".join(str(element.value) for element in app.sidebar.json)
    assert "Nimbus Portal" in sidebar_json
    assert "React" in sidebar_json
    assert any("2 messages" in caption.value for caption in app.sidebar.caption)


def test_the_form_has_a_transcript_field(api: FakeApi) -> None:
    app = _app()

    assert app.text_area(key="transcript_0").label == "Transcript"


def test_new_conversation_creates_another_session_and_resets_the_state(api: FakeApi) -> None:
    app = _app()
    _submit(app)
    assert app.session_state["result_body"]["text"] == "RESULT_TEXT"

    app.button(key="new_conversation").click().run()

    assert app.session_state["session_id"] == "s-2"
    assert api.created == 2
    assert "result_body" not in app.session_state
    assert not any("RESULT_TEXT" in markdown.value for markdown in app.markdown)


def test_submit_posts_the_form_to_the_session_endpoint_and_shows_the_result(api: FakeApi) -> None:
    app = _app()

    _submit(app)

    call = api.estimate_calls[0]
    assert call["url"] == f"{BASE}/sessions/s-1/estimate"
    assert call["data"]["transcript"] == TRANSCRIPT
    assert call["data"]["detail_level"] == "summary"
    assert call["files"] is None
    assert any("RESULT_TEXT" in markdown.value for markdown in app.markdown)
    assert not app.error


def test_the_form_is_cleared_after_a_successful_turn(api: FakeApi) -> None:
    app = _app()

    _submit(app)

    assert app.text_area(key=f"transcript_{app.session_state['form_version']}").value == ""


def test_a_short_transcript_is_rejected_before_calling_the_service(api: FakeApi) -> None:
    app = _app()

    _submit(app, "too short")

    assert api.estimate_calls == []
    assert any("at least 20" in error.value for error in app.error)


def test_an_expired_session_asks_for_a_new_conversation(api: FakeApi) -> None:
    api.estimate_response = (404, {"detail": "session_not_found"})
    app = _app()

    _submit(app)

    assert any("no longer exists" in error.value for error in app.error)


def test_service_errors_show_the_status_and_the_detail(api: FakeApi) -> None:
    api.estimate_response = (415, {"detail": {"filename": "a.txt", "message": "Unsupported attachment type"}})
    app = _app()

    _submit(app)

    assert any("415" in error.value and "Unsupported attachment type" in error.value for error in app.error)


def test_a_502_shows_the_reason_code_and_message(api: FakeApi) -> None:
    api.estimate_response = (
        502,
        {
            "detail": "The LLM provider failed to generate the estimation.",
            "reason": {
                "code": "validation_failed",
                "message": "phases sum (1 EUR) does not match total_cost_eur (2 EUR)",
            },
        },
    )
    app = _app()

    _submit(app)

    shown = " ".join(error.value for error in app.error)
    assert "502" in shown
    assert "validation_failed" in shown
    assert "phases sum (1 EUR)" in shown
