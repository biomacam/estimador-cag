---
name: add-estimation-endpoint
description: 'Use when adding a new API endpoint or router to the estimator FastAPI service, following its layered architecture (router -> schema -> service -> context/config). Triggers: "add endpoint", "new route", "new router", "expose API".'
---

# Add Estimation Endpoint (layered architecture)

## When to Use
- Adding a new HTTP endpoint to the estimator service beyond `/api/v1/estimate` and `/health`.

## Existing Layering (must be respected)
1. **Router** (`app/routers/`) — FastAPI `APIRouter`, request/response only via schemas, translates domain exceptions to `HTTPException`. No business logic.
2. **Schema** (`app/schemas/`) — Pydantic `BaseModel`s defining the request/response contract (`Field(..., description=...)`), validated at the boundary.
3. **Service** (`app/services/`) — business logic, orchestrates calls to `context/` and `config`, raises a dedicated `<X>ServiceError` exception on failure.
4. **Context/Config** (`app/context/`, `app/config.py`) — static knowledge base and environment-driven settings (`get_settings()`, `lru_cache`d singleton).

## Procedure
1. Define request/response models in `app/schemas/<name>.py` (mirror `EstimationRequest`/`EstimationResponse` in `estimator/app/schemas/estimation.py`).
2. Implement the logic in `app/services/<name>_service.py`: define a module-level `<Name>ServiceError(Exception)`, use `structlog.get_logger()` for `log.info`/`log.error`, and read settings via `get_settings()` — never read env vars directly.
3. Create `app/routers/<name>.py` with `router = APIRouter(prefix="/api/v1", tags=["<name>"])`, a handler that calls the service and catches its `ServiceError` to raise `HTTPException(status_code=500, detail=str(exc))`.
4. Register the router in `estimator/app/main.py` via `app.include_router(<name>.router)`.
5. Add tests under `estimator/tests/` following the `client` fixture in `conftest.py`.
6. Update the architecture diagram/table in `estimator/README.md` if a new layer or external dependency is introduced.

## FastAPI convention: `async def` vs `def` in route handlers
`async def` only helps when the body actually `await`s non-blocking I/O. The current `create_estimation` handler in `app/routers/estimations.py` is declared `async def` but calls the fully synchronous `generate_estimation()` (blocking OpenAI/Anthropic SDK calls, no `await`) — this runs the blocking call directly on the event loop instead of FastAPI's threadpool, which can stall other concurrent requests.
- If the service call chain stays synchronous/blocking, declare the handler as plain `def` — FastAPI will run it in an external threadpool automatically.
- Only use `async def` once the underlying service is truly async (e.g. using `AsyncOpenAI`/`AsyncAnthropic` with real `await` calls).

## Input validation for any field forwarded to an LLM prompt
Any request schema field whose value ends up inside a prompt (like `EstimationRequest.transcription`) is effectively "the bill" for that request — an unbounded field lets a client generate an arbitrarily expensive or context-window-breaking call. Every such field MUST have both bounds, not just a minimum:
```python
transcription: str = Field(..., min_length=50, max_length=50_000)
```
`min_length` alone (or no bound at all) only prevents a useless empty call; it does nothing to cap cost. Pydantic rejects out-of-range input with a 422 before the service layer runs, so no LLM call — and no token cost — is incurred for invalid input.

When that field's raw text is interpolated into the `user` message sent to the LLM (see `build_system_prompt`/`generate_estimation` in `services/llm_service.py`), treat it as untrusted data, not as part of the instructions:
- Wrap it with a delimiter and tell the model explicitly that instructions inside the delimiter don't override the system prompt.
- A per-request unpredictable delimiter (e.g. `secrets.token_hex(8)`) is more robust than a fixed tag like `<transcripcion>`, since fixed tags can be closed early by attacker-controlled input that happens to contain the same closing tag.
