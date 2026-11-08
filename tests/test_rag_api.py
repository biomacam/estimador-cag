"""HTTP-level tests for the RAG routers: auth boundary, rate limiting, request id,
stage endpoints, runtime retrieval config and the idempotency store.

Downstream work (LLM, embeddings, pgvector) is stubbed; the real context assembler,
citation validation and coherence check run.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.rag.security as security
import app.routers.rag_estimate as estimate_router
import app.routers.rag_estimate_stages as stages
import app.routers.rag_retrieval as retrieval_router
from app.config import get_settings
from app.dependencies import get_runtime_retrieval_config
from app.main import app
from app.rag.idempotency import IdempotencyStore
from app.rag.runtime_config import RuntimeRetrievalConfig
from app.rag.schemas import (
    Estimate,
    EstimationQuery,
    RetrievalResult,
    RetrievedChunk,
    SourceCitation,
)

EST_KEY = "estimate-secret"
RET_KEY = "retrieval-secret"
_TRANSCRIPT = {"transcript": "x" * 200}
_QUERY = {"query_text": "online store card checkout"}


def _chunk(cid: int, content: str = "Auth & RBAC component: ~12 engineer-days.") -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, content=content, sector="ecommerce", project_year=2024,
        chunk_type="budget_component", distance=0.3,
    )


def _h(key: str = EST_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


class _Embedder:
    @staticmethod
    def embed_one(text):
        return [0.0] * 1536


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(RETRIEVAL_API_KEY=RET_KEY, ESTIMATE_API_KEY=EST_KEY),
    )

    async def fake_reformulate(transcript):
        return EstimationQuery(function="online store with card checkout", sector="ecommerce")

    async def fake_search(query_embedding, **kwargs):
        return RetrievalResult(chunks=[_chunk(1), _chunk(2)], low_confidence=False, candidates_evaluated=12)

    async def fake_retrieve(**kwargs):
        return RetrievalResult(chunks=[_chunk(1)], low_confidence=False, candidates_evaluated=3)

    async def fake_estimate(transcript, idempotency_key=None):
        return Estimate(confidence="insufficient", reasoning="stub", insufficient_context_explanation="stub")

    runtime = SimpleNamespace(effective_search_mode=lambda: "vector", effective_rerank=lambda: False)

    monkeypatch.setattr(stages, "reformulate_query", fake_reformulate)
    monkeypatch.setattr(stages, "compose_search_text", lambda q: "online store card checkout ecommerce")
    monkeypatch.setattr(stages, "get_embedder", lambda: _Embedder())
    monkeypatch.setattr(stages, "search_chunks", fake_search)
    monkeypatch.setattr(retrieval_router, "get_embedder", lambda: _Embedder())
    monkeypatch.setattr(retrieval_router, "get_runtime_retrieval_config", lambda: runtime)
    monkeypatch.setattr(retrieval_router, "retrieve", fake_retrieve)
    monkeypatch.setattr(estimate_router, "estimate_from_transcript", fake_estimate)


@pytest.fixture
def api() -> TestClient:
    return TestClient(app)


# --- auth boundary ---------------------------------------------------------


@pytest.mark.parametrize(
    "path,body,own_key,other_key",
    [
        ("/v1/retrieval/search", _QUERY, RET_KEY, EST_KEY),
        ("/v1/estimate/from-transcript", _TRANSCRIPT, EST_KEY, RET_KEY),
        ("/v1/estimate/stages/reformulate", _TRANSCRIPT, EST_KEY, RET_KEY),
        ("/v1/estimate/stages/retrieve", _QUERY, EST_KEY, RET_KEY),
        ("/v1/estimate/stages/assemble", {"chunks": []}, EST_KEY, RET_KEY),
    ],
)
def test_each_router_accepts_only_its_own_key(api, path, body, own_key, other_key):
    assert api.post(path, json=body).status_code == 401
    assert api.post(path, json=body, headers=_h("wrong")).status_code == 401
    assert api.post(path, json=body, headers=_h(other_key)).status_code == 401
    assert api.post(path, json=body, headers=_h(own_key)).status_code == 200


def test_blank_configured_key_disables_the_router(api, monkeypatch):
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(RETRIEVAL_API_KEY=None, ESTIMATE_API_KEY=None)
    )
    assert api.post("/v1/retrieval/search", json=_QUERY, headers=_h("")).status_code == 401


def test_invalid_payload_is_a_422_after_auth(api):
    r = api.post("/v1/estimate/from-transcript", json={"transcript": "too short"}, headers=_h())
    assert r.status_code == 422


# --- rate limiting + request id --------------------------------------------


def test_estimate_is_limited_to_10_per_minute_per_api_key(api, monkeypatch):
    key = f"rate-{uuid.uuid4()}"
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(RETRIEVAL_API_KEY=key, ESTIMATE_API_KEY=key)
    )

    statuses = [
        api.post("/v1/estimate/from-transcript", json=_TRANSCRIPT, headers=_h(key)).status_code
        for _ in range(11)
    ]

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


def test_rate_limit_response_has_retry_after(api, monkeypatch):
    key = f"rate-{uuid.uuid4()}"
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(RETRIEVAL_API_KEY=key, ESTIMATE_API_KEY=key)
    )
    for _ in range(10):
        api.post("/v1/estimate/from-transcript", json=_TRANSCRIPT, headers=_h(key))

    r = api.post("/v1/estimate/from-transcript", json=_TRANSCRIPT, headers=_h(key))

    assert r.status_code == 429
    assert r.headers["Retry-After"] == "60"
    assert r.json()["detail"] == "Rate limit exceeded."


def test_request_id_header_is_generated_and_echoed(api):
    generated = api.post("/v1/retrieval/search", json=_QUERY, headers=_h(RET_KEY))
    assert generated.headers["X-Request-ID"]

    echoed = api.post(
        "/v1/retrieval/search", json=_QUERY, headers={**_h(RET_KEY), "X-Request-ID": "trace-42"}
    )
    assert echoed.headers["X-Request-ID"] == "trace-42"


# --- retrieval router ------------------------------------------------------


def test_retrieval_resolves_mode_from_request_then_runtime(api, monkeypatch):
    seen: dict = {}

    async def spy(**kwargs):
        seen.update(kwargs)
        return RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=0)

    monkeypatch.setattr(retrieval_router, "retrieve", spy)

    r = api.post("/v1/retrieval/search", json=_QUERY, headers=_h(RET_KEY))
    assert r.status_code == 200 and r.json()["low_confidence"] is True
    assert (seen["search_mode"], seen["rerank"]) == ("vector", False)  # runtime defaults

    api.post(
        "/v1/retrieval/search",
        json={**_QUERY, "search_mode": "hybrid", "rerank": True},
        headers=_h(RET_KEY),
    )
    assert (seen["search_mode"], seen["rerank"]) == ("hybrid", True)  # request wins


def test_retrieval_failure_maps_to_502(api, monkeypatch):
    from app.rag.errors import RetrievalError

    async def boom(**kwargs):
        raise RetrievalError("db down")

    monkeypatch.setattr(retrieval_router, "retrieve", boom)
    assert api.post("/v1/retrieval/search", json=_QUERY, headers=_h(RET_KEY)).status_code == 502


def test_estimate_pipeline_failure_maps_to_502(api, monkeypatch):
    from app.rag.errors import GenerationError

    async def boom(transcript, idempotency_key=None):
        raise GenerationError("llm down")

    monkeypatch.setattr(estimate_router, "estimate_from_transcript", boom)
    r = api.post("/v1/estimate/from-transcript", json=_TRANSCRIPT, headers=_h())
    assert r.status_code == 502


# --- stage endpoints -------------------------------------------------------


def test_reformulate_stage_returns_brief_and_search_text(api):
    r = api.post("/v1/estimate/stages/reformulate", json=_TRANSCRIPT, headers=_h())
    assert r.status_code == 200
    assert r.json()["query"]["sector"] == "ecommerce"
    assert r.json()["search_text"] == "online store card checkout ecommerce"


def test_retrieve_stage_passes_chunks_and_soft_fail_through(api, monkeypatch):
    ok = api.post("/v1/estimate/stages/retrieve", json=_QUERY, headers=_h()).json()
    assert len(ok["chunks"]) == 2 and ok["low_confidence"] is False

    async def empty(query_embedding, **kwargs):
        return RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=9)

    monkeypatch.setattr(stages, "search_chunks", empty)
    soft = api.post("/v1/estimate/stages/retrieve", json=_QUERY, headers=_h()).json()
    assert soft["chunks"] == [] and soft["low_confidence"] is True


def test_assemble_stage_wraps_chunks_in_xml(api):
    payload = {"chunks": [_chunk(1).model_dump(), _chunk(2).model_dump()]}
    body = api.post("/v1/estimate/stages/assemble", json=payload, headers=_h()).json()

    assert '<source id="1"' in body["context_block"]
    assert body["dropped_count"] == 0
    assert body["token_count"] > 0
    assert len(body["kept_chunks"]) == 2


def test_assemble_stage_drops_whole_chunks_over_budget(api):
    big = [_chunk(i, content="word " * 300).model_dump() for i in range(1, 6)]
    body = api.post(
        "/v1/estimate/stages/assemble",
        json={"chunks": big, "max_context_tokens": 300},
        headers=_h(),
    ).json()

    assert body["dropped_count"] > 0
    assert len(body["kept_chunks"]) + body["dropped_count"] == 5


def _generate_payload() -> dict:
    return {
        "context_block": '<source id="1">...</source>',
        "query": EstimationQuery(function="online store").model_dump(),
        "kept_chunks": [_chunk(1).model_dump()],
    }


def test_generate_stage_reports_fabricated_citations(api, monkeypatch):
    estimate = Estimate(
        confidence="high",
        reasoning="r",
        sources=[
            SourceCitation(source_id=1, relevance="primary", used_for="auth"),
            SourceCitation(source_id=999, relevance="supporting", used_for="ghost"),
        ],
    )

    async def fake_generate(context_block, structured_query):
        return estimate

    monkeypatch.setattr(stages, "generate_estimate", fake_generate)
    body = api.post("/v1/estimate/stages/generate", json=_generate_payload(), headers=_h()).json()

    assert body["fabricated_source_ids"] == [999]
    assert body["coherent"] is True


def test_generate_stage_reports_incoherent_insufficient_estimate(api, monkeypatch):
    estimate = Estimate(
        confidence="insufficient",
        reasoning="r",
        total_engineer_days=40,
        insufficient_context_explanation="should not have numbers",
    )

    async def fake_generate(context_block, structured_query):
        return estimate

    monkeypatch.setattr(stages, "generate_estimate", fake_generate)
    body = api.post("/v1/estimate/stages/generate", json=_generate_payload(), headers=_h()).json()

    assert body["coherent"] is False


# --- runtime retrieval config endpoint -------------------------------------


@pytest.fixture
def runtime_config():
    config = RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), get_settings())
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: config
    yield config
    app.dependency_overrides.pop(get_runtime_retrieval_config, None)


def test_config_endpoint_flips_and_resets_the_toggles(api, runtime_config):
    before = api.get("/api/v1/config/retrieval").json()["retrieval"]
    assert before["RETRIEVAL_SEARCH_MODE"]["overridden"] is False

    r = api.put("/api/v1/config/retrieval", json={"search_mode": "hybrid", "rerank": True})
    assert r.status_code == 200
    after = r.json()["retrieval"]
    assert after["RETRIEVAL_SEARCH_MODE"] == {
        "effective": "hybrid", "default": before["RETRIEVAL_SEARCH_MODE"]["default"], "overridden": True,
    }
    assert after["RERANKER_ENABLED"]["effective"] is True
    assert runtime_config.effective_search_mode() == "hybrid"

    # An explicit null resets one knob; the absent key leaves the other untouched.
    reset = api.put("/api/v1/config/retrieval", json={"search_mode": None}).json()["retrieval"]
    assert reset["RETRIEVAL_SEARCH_MODE"]["overridden"] is False
    assert reset["RERANKER_ENABLED"]["overridden"] is True


def test_config_endpoint_rejects_an_unknown_search_mode(api, runtime_config):
    assert api.put("/api/v1/config/retrieval", json={"search_mode": "fuzzy"}).status_code == 422


# --- idempotency store -----------------------------------------------------


def _estimate() -> Estimate:
    return Estimate(confidence="insufficient", reasoning="r", insufficient_context_explanation="x")


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_idempotency_store_roundtrip_and_delete(backend):
    client = fakeredis.FakeRedis(decode_responses=True) if backend == "redis" else None
    store = IdempotencyStore(redis_client=client, ttl=60)

    assert store.get("k") is None
    store.set("k", _estimate())
    assert store.get("k") == _estimate()
    store.delete("k")
    assert store.get("k") is None


def test_idempotency_store_expires_in_memory_entries(monkeypatch):
    import app.rag.idempotency as module

    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    store = IdempotencyStore(ttl=10)
    store.set("k", _estimate())

    now[0] += 11
    assert store.get("k") is None


def test_idempotency_store_treats_a_corrupt_redis_entry_as_a_miss():
    client = fakeredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis_client=client, ttl=60)
    client.set("idempotency:estimate:k", "{not json")

    assert store.get("k") is None
    assert client.get("idempotency:estimate:k") is None
