---
name: test-estimator-service
description: 'Use when writing or running pytest tests for the estimator FastAPI service (routers, services, schemas), or mocking LLM provider calls in tests. Triggers: "write a test", "run tests", "pytest", "mock LLM call", "TestClient".'
---

# Test Estimator Service

## When to Use
- Adding test coverage for routers/services in `estimator/`.
- Running the existing test suite.
- Mocking OpenAI/Anthropic calls so tests don't hit real APIs.

## Existing Conventions
- `estimator/tests/conftest.py` exposes a `client` fixture (`fastapi.testclient.TestClient` wrapping `app.main.app`).
- Test files: `tests/test_*.py`, using plain `def test_...(client: TestClient) -> None`.
- `pyproject.toml` sets `asyncio_mode = "auto"` (pytest-asyncio) and `testpaths = ["tests"]`.
- Dev dependencies (`pytest`, `pytest-asyncio`, `httpx`, `ruff`) live in the `dev` dependency group.

## Procedure
1. Run the full suite from the `estimator/` directory:
   ```bash
   uv run pytest
   ```
2. For a new router/service, add `tests/test_<name>.py` using the `client` fixture, following `tests/test_health.py` as the minimal template.
3. To avoid real LLM calls, monkeypatch the service entry point, e.g.:
   ```python
   def test_create_estimation(client, monkeypatch):
       monkeypatch.setattr(
           "app.routers.estimations.generate_estimation",
           lambda transcription: {
               "estimation": "## Fake\n...",
               "model": "gpt-4o-mini",
               "provider": "openai",
               "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
           },
       )
       response = client.post("/api/v1/estimate", json={"transcription": "x" * 60})
       assert response.status_code == 200
   ```
4. To test provider-level failures, monkeypatch `_call_openai`/`_call_anthropic` to raise, and assert the router returns HTTP 500 with the `LLMServiceError` message.
5. Keep assertions on `response.json()` structure aligned with the Pydantic schemas in `app/schemas/`.

## Rule: assert on content/invariants, not just on status codes
A test that only checks `status_code == 200` (or `response.json() is not None`) can pass even when the thing the code is actually supposed to guarantee is broken — "it responded successfully" is not the same as "it did the right thing". This is the general pattern behind the two required tests below, and applies to any future test in this service:
- If a function's job is to **produce/inject specific content** (e.g. put the CAG examples in a prompt), assert that content is present — not just that the function returned *something*.
- If a code path's job is to **withhold information** (e.g. not leak an internal error message), assert that information is absent from the output — not just that the status code is the expected one.
- Ask "what invariant would silently break while every status code stays green?" before considering a test complete.

## Required tests — don't skip these
Two invariants have historically gone untested and are the ones most worth protecting:

1. **The CAG invariant**: the reference examples actually reach the system prompt. A test that only checks `status_code == 200` passes even if `ESTIMATION_EXAMPLES` is never imported by `build_system_prompt()` — i.e. even if CAG doesn't exist. Always include:
   ```python
   def test_examples_reach_the_system_prompt():
       from app.services.llm_service import build_system_prompt
       from app.context.examples import ESTIMATION_EXAMPLES

       prompt = build_system_prompt()
       for example in ESTIMATION_EXAMPLES:
           assert example["estimation"] in prompt
   ```
2. **No secret/internal-detail leakage on provider failure**: assert an internal exception message does NOT appear in the HTTP response body, not just that the status code is correct:
   ```python
   def test_provider_error_does_not_leak_details(client, monkeypatch):
       def boom(*args, **kwargs):
           raise LLMServiceError("sk-proj-super-secret-key-fragment")
       monkeypatch.setattr("app.routers.estimations.generate_estimation", boom)

       response = client.post("/api/v1/estimate", json={"transcription": "x" * 60})

       assert response.status_code == 500
       assert "sk-proj" not in response.text
   ```
   If this test currently fails, the router/service is forwarding `str(exc)` straight to the client — see the Safety Checklist in the `add-llm-provider` skill.
