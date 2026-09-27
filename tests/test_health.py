from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_returns_200_even_without_any_api_key_configured() -> None:
    """The regression this guards against: Settings() used to raise when no API key
    was set, which killed the app before /health could respond at all."""
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None
    )
    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["llm_configured"] is False
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_health_reports_llm_configured_true_when_key_present() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY=None
    )
    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.json()["llm_configured"] is True
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_no_cors_headers_are_returned_for_cross_origin_requests(client: TestClient) -> None:
    """No browser frontend calls this API cross-origin (Streamlit calls it
    server-side; the SSE demo page fetches a same-origin relative path), so
    there must be no CORS middleware reflecting arbitrary Origins back.
    """
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {h.lower() for h in response.headers}
