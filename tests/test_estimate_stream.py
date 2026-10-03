from collections.abc import Iterator

from fastapi.testclient import TestClient
from litellm.exceptions import APIError

from app.dependencies import get_llm_wrapper
from app.main import app
from app.services.llm_service import LLM_PROVIDER_ERROR_MESSAGE
from app.services.llm_wrapper import LLMConfigurationError, LLMTruncatedResponseError
from tests.helpers import unwrap_untrusted_input


class _StubWrapper:
    """Minimal stand-in for LLMWrapper that yields canned chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[dict] = []

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None,
        max_tokens: int,
    ) -> Iterator[str]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model_override": model_override,
                "max_tokens": max_tokens,
            }
        )
        for chunk in self._chunks:
            yield chunk


def test_stream_endpoint_emits_token_and_done_events() -> None:
    stub = _StubWrapper(chunks=["Hello ", "from ", "the ", "estimator."])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        # The response is one or more SSE messages separated by blank lines.
        assert "event: token" in body
        assert "data: Hello" in body
        assert "data: estimator." in body
        assert "event: done" in body
        assert len(stub.calls) == 1
        framed_user_message = stub.calls[0]["user_message"]
        assert unwrap_untrusted_input(framed_user_message) == "x" * 60
        # The system prompt must reference the same tag the user message is
        # wrapped in, otherwise the model has no way to know where the data starts.
        tag = framed_user_message.split("<user-data-", 1)[1].split(">", 1)[0]
        assert f"<user-data-{tag}>" in stub.calls[0]["system_prompt"]
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_serialises_multiline_chunk_as_multiple_data_lines() -> None:
    """A chunk with internal newlines must be split into one ``data:`` line per
    physical line, with the message terminated by a blank line. Clients must
    join those data lines with ``\\n`` to recover the original payload — that
    is the contract this test pins down on the server side.
    """
    multiline_chunk = "## Project summary\n\nThis is the body."
    stub = _StubWrapper(chunks=[multiline_chunk])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert "data: ## Project summary" in body
        assert "data: This is the body." in body
        assert "event: token" in body
        assert "event: done" in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_rejects_short_transcription() -> None:
    stub = _StubWrapper(chunks=["irrelevant"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/estimate/stream",
                json={"transcription": "too short"},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_rejects_transcription_over_50000_tokens() -> None:
    stub = _StubWrapper(chunks=["should not be reached"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/estimate/stream",
                json={"transcription": "hello " * 50_001},
            )
        assert response.status_code == 422
        assert "50,000 tokens" in response.text
        assert stub.calls == []
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_appends_annual_maintenance_when_total_cost_present() -> None:
    stub = _StubWrapper(chunks=["- **Total cost:** 10,000 EUR"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert "Annual maintenance" in body
        assert "1,200.00 EUR" in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


class _TruncatingStubWrapper:
    """Stub that yields some chunks then raises, simulating a provider cutoff."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None,
        max_tokens: int,
    ) -> Iterator[str]:
        yield from self._chunks
        raise LLMTruncatedResponseError(
            "Provider stopped before completing the response (finish_reason='length')"
        )


def test_stream_endpoint_emits_error_event_without_done_when_truncated() -> None:
    stub = _TruncatingStubWrapper(chunks=["Hello ", "wor"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert "event: token" in body
        assert "event: error" in body
        assert "event: done" not in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


class _FailingStubWrapper:
    """Stub whose complete_stream raises a real provider exception mid-stream."""

    def __init__(self, chunks: list[str], exc: Exception) -> None:
        self._chunks = chunks
        self._exc = exc

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None,
        max_tokens: int,
    ) -> Iterator[str]:
        yield from self._chunks
        raise self._exc


def test_stream_endpoint_sanitizes_provider_error_event() -> None:
    secret_fragment = "sk-proj-super-secret-fragment"
    stub = _FailingStubWrapper(
        chunks=["Hello "],
        exc=APIError(
            status_code=401,
            message=secret_fragment,
            llm_provider="openai",
            model="gpt-4o-mini",
        ),
    )
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert secret_fragment not in body
        assert LLM_PROVIDER_ERROR_MESSAGE in body
        assert "event: error" in body
        assert "event: done" not in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_returns_503_when_llm_not_configured() -> None:
    def raise_not_configured() -> None:
        raise LLMConfigurationError(
            "No LLM provider is configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    app.dependency_overrides[get_llm_wrapper] = raise_not_configured
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/estimate/stream",
                json={"transcription": "x" * 60},
            )
        assert response.status_code == 503
        assert "OPENAI_API_KEY" in response.text
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)
