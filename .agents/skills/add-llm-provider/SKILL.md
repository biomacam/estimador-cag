---
name: add-llm-provider
description: 'Use when adding support for a new LLM provider (e.g. Gemini, Mistral, Azure OpenAI) to the estimator service, or when modifying how LLM_PROVIDER is selected/validated. Triggers: "add provider", "new LLM", "support Gemini/Mistral", "switch LLM provider".'
---

# Add LLM Provider (estimator)

## When to Use
- Adding a new LLM provider alongside the existing OpenAI/Anthropic support.
- Changing the provider selection or API-key validation logic.

## Existing Pattern
- `estimator/app/config.py`: `Settings.LLM_PROVIDER` is a `Literal["openai", "anthropic"]`; `validate_api_key_for_provider` (a `model_validator`) enforces that the matching `*_API_KEY` is set.
- `estimator/app/services/llm_service.py`: `generate_estimation()` dispatches on `settings.LLM_PROVIDER` to a private `_call_<provider>()` function. Each function lazily imports its SDK inside the function body (keeps optional deps out of module import time) and returns a dict shaped like:
  ```python
  {
      "estimation": str,
      "model": str,
      "provider": str,
      "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
  }
  ```
- Provider-agnostic failures are wrapped as `LLMServiceError` and surfaced by the router as HTTP 500.

## Procedure
1. In `app/config.py`: extend the `LLM_PROVIDER` `Literal` with the new value, add the corresponding `<PROVIDER>_API_KEY: str | None = None` field, and add a branch in `validate_api_key_for_provider`.
2. Add the provider's SDK to `dependencies` in `pyproject.toml`.
3. In `app/services/llm_service.py`, add a `_call_<provider>(...)` function mirroring `_call_openai`/`_call_anthropic`: build the request, call `structlog` `log.info("llm_response_received", provider=..., input_tokens=..., output_tokens=...)`, and return the same dict shape.
4. Add a branch for the new provider in `generate_estimation()`'s if/elif dispatch (keep the existing `try/except` wrapping into `LLMServiceError`).
5. Document the new env var in `estimator/README.md` and `.env.example`.
6. Add/update a test that monkeypatches the new `_call_<provider>` to avoid real API calls (see the `test-estimator-service` skill).
7. Run every new/changed `_call_<provider>` through the **Safety Checklist** below before considering it done.

## Safety Checklist for any `_call_<provider>` implementation
These are recurring mistakes found in past reviews. Every SDK client construction and response handling MUST satisfy all of these:

1. **Always set `timeout` and `max_retries` explicitly when constructing the client.** SDK defaults are unsafe (e.g. OpenAI's default is 600s with 2 retries — a single slow request can hold a worker for ~30 minutes and starve the service). Never call `OpenAI(api_key=...)` or `Anthropic(api_key=...)` without both:
   ```python
   client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0, max_retries=2)
   client = Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=30.0, max_retries=2)
   ```
   Prefer reading the timeout from `Settings` (e.g. `settings.LLM_TIMEOUT_SECONDS`) instead of a bare literal, so it can be tuned without a code change.
2. **Always check for truncation before returning `.content`.** A `max_tokens` cap means the response can be cut mid-output with no exception raised — reading only `.content`/`.text` silently returns an incomplete result as if it were a normal 200. After the call, check the finish signal and raise `LLMServiceError` if truncated:
   ```python
   # OpenAI
   if response.choices[0].finish_reason == "length":
       raise LLMServiceError(f"Response truncated at {MAX_TOKENS} tokens.")
   # Anthropic
   if response.stop_reason == "max_tokens":
       raise LLMServiceError(f"Response truncated at {MAX_TOKENS} tokens.")
   ```
3. **Never let a raw provider exception message reach the HTTP client.** Catch specific SDK exception classes (`RateLimitError`, `APIConnectionError`, `APIStatusError`, etc.), not bare `Exception` — a bare `except Exception` also swallows your own bugs and reports them as "LLM provider failure". Log the full exception server-side with `structlog` and raise `LLMServiceError` with a fixed, generic message (never `str(exc)`, which can contain API key fragments, org IDs, or internal URLs):
   ```python
   except (RateLimitError, APIConnectionError, APIStatusError) as exc:
       log.error("llm_provider_failed", provider="openai", error=str(exc))
       raise LLMServiceError("No se pudo generar la estimación.") from exc
   ```
4. **Don't reintroduce independently-defaulted provider/model fields.** `LLM_PROVIDER`/`LLM_MODEL` used to exist as separate fields and were removed because nothing enforced they stayed consistent with each other. Model selection is `PRIMARY_MODEL`/`FALLBACK_MODEL` (`app/config.py`) plus the per-request `model` override; `LLMWrapper._api_key_for()` (`app/services/llm_wrapper.py`) derives the provider — and therefore the right API key — from the model string itself via `_provider_from_model()`, for both the primary and fallback deployments. When adding a provider, extend `_provider_from_model()`'s prefix matching rather than adding a new coupled-by-convention field pair.
