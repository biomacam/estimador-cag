"""Flow tests for the RAG pipeline: reformulation, hybrid retrieval and the orchestrator.

Every collaborator (LLM, embedder, pgvector, cross-encoder) is faked: the point is
the wiring — which stage runs, in what order, and the soft-fail / retry contracts.
Async code runs through ``asyncio.run`` (no pytest-asyncio in this project).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import app.dependencies as deps
import app.persistence.database as database
from app.rag import estimator as orch
from app.rag import query_reformulator as qr
from app.rag.errors import MalformedEstimateError, ReformulationError
from app.rag.retrieval.pipeline import retrieve
from app.rag.schemas import (
    Estimate,
    EstimationQuery,
    RetrievalResult,
    RetrievedChunk,
    SourceCitation,
    TaskItem,
    WorkModule,
)

_SETTINGS = SimpleNamespace(
    REFORMULATION_MODEL="gpt-5-mini",
    GENERATION_MODEL="gpt-5",
    GENERATION_REASONING_EFFORT="high",
    GENERATION_MAX_TOKENS=64_000,
    RETRIEVAL_TOP_K=10,
    RETRIEVAL_DISTANCE_THRESHOLD=0.6,
    MAX_CONTEXT_TOKENS=100_000,
    RETRIEVAL_RECALL_TOP_K=50,
    RERANK_TOP_N=5,
    RRF_K=60,
)


class CharEncoder:
    def encode(self, text: str) -> list[str]:
        return list(text)


# --- query reformulation ---------------------------------------------------


class FakeWrapper:
    def __init__(self, *, structured=None, structured_error=None, completion=None):
        self._structured = structured
        self._structured_error = structured_error
        self._completion = completion
        self.structured_calls = 0
        self.completion_calls = 0

    def complete_structured(self, **kwargs):
        self.structured_calls += 1
        if self._structured_error is not None:
            raise self._structured_error
        return self._structured, {"model": "gpt-5-mini"}

    def complete(self, **kwargs):
        self.completion_calls += 1
        return self._completion


@pytest.fixture
def reformulation_settings(monkeypatch):
    monkeypatch.setattr(qr, "get_settings", lambda: _SETTINGS)


def test_reformulate_query_happy_path(monkeypatch, reformulation_settings):
    expected = EstimationQuery(function="B2B payments marketplace", sector="finance")
    fake = FakeWrapper(structured=expected)
    monkeypatch.setattr(deps, "get_llm_wrapper", lambda: fake)

    assert asyncio.run(qr.reformulate_query("long messy transcript")) == expected
    assert (fake.structured_calls, fake.completion_calls) == (1, 0)


def test_reformulate_query_falls_back_to_free_text(monkeypatch, reformulation_settings):
    fake = FakeWrapper(
        structured_error=RuntimeError("schema validation failed"),
        completion={"estimation": "ecommerce storefront with card checkout"},
    )
    monkeypatch.setattr(deps, "get_llm_wrapper", lambda: fake)

    result = asyncio.run(qr.reformulate_query("transcript"))

    assert result.function == "ecommerce storefront with card checkout"
    assert result.technologies == []
    assert (fake.structured_calls, fake.completion_calls) == (1, 1)


def test_reformulate_query_raises_when_both_paths_fail(monkeypatch, reformulation_settings):
    fake = FakeWrapper(structured_error=RuntimeError("boom"), completion={"estimation": "   "})
    monkeypatch.setattr(deps, "get_llm_wrapper", lambda: fake)

    with pytest.raises(ReformulationError):
        asyncio.run(qr.reformulate_query("transcript"))


# --- hybrid retrieval ------------------------------------------------------


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _row(chunk_id: int, *, distance: float = 0.3):
    return SimpleNamespace(
        id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"chunk-{chunk_id}",
        metadata_={"client_sector": "ecommerce", "year": 2024, "budget_id": f"BUD-{chunk_id}"},
        distance=distance,
    )


class FakeStore:
    def __init__(self, vector_rows, lexical_rows, candidates=12):
        self.vector_rows = vector_rows
        self.lexical_rows = lexical_rows
        self.candidates = candidates
        self.vector_calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    async def search_filtered(self, session, **kwargs):
        self.vector_calls.append(kwargs)
        return self.vector_rows[: kwargs["top_k"]], self.candidates

    async def search_lexical(self, session, **kwargs):
        self.lexical_calls.append(kwargs)
        return self.lexical_rows[: kwargs["top_k"]]


class FakeReranker:
    def __init__(self, scores_by_content):
        self.scores_by_content = scores_by_content

    def rerank(self, query, candidates, *, top_n, text_of=lambda c: c.content):
        ranked = sorted(candidates, key=lambda c: self.scores_by_content[text_of(c)], reverse=True)
        return ranked[:top_n]


@pytest.fixture
def wire_store(monkeypatch):
    def _wire(store):
        monkeypatch.setattr(database, "get_async_session_factory", lambda: FakeSession)
        monkeypatch.setattr(deps, "get_chunk_store", lambda: store)

    return _wire


def _retrieve(**kwargs):
    return asyncio.run(
        retrieve(query_embedding=[0.0] * 1536, query_text="ecommerce checkout", **kwargs)
    )


def test_vector_mode_without_rerank_recalls_exactly_top_k(wire_store):
    store = FakeStore(vector_rows=[_row(1), _row(2), _row(3)], lexical_rows=[])
    wire_store(store)

    result = _retrieve(search_mode="vector", rerank=False, top_k=2)

    assert [c.id for c in result.chunks] == [1, 2]
    assert result.low_confidence is False
    assert result.candidates_evaluated == 12
    assert result.chunks[0].budget_id == "BUD-1"
    assert store.vector_calls[0]["top_k"] == 2
    assert store.lexical_calls == []


def test_hybrid_mode_fuses_both_branches_and_recalls_wide(wire_store):
    store = FakeStore(vector_rows=[_row(1), _row(2)], lexical_rows=[_row(3), _row(1)])
    wire_store(store)

    result = _retrieve(search_mode="hybrid", rerank=False, top_k=5, recall_k=50)

    assert [c.id for c in result.chunks] == [1, 3, 2]
    assert result.chunks[0].distance == 0.3  # vector distance wins for shared hits
    assert result.chunks[1].distance == 1.0  # lexical-only display sentinel
    assert store.vector_calls[0]["top_k"] == 50
    assert store.lexical_calls[0]["top_k"] == 50


@pytest.mark.parametrize(
    "vector_ids,lexical_ids,expected",
    [
        ([1, 2], [3, 2], [2, 1, 3]),
        ([], [3, 2], [3, 2]),
        ([1, 2], [], [1, 2]),
        ([], [], []),
    ],
)
def test_hybrid_returns_a_unique_fused_ranking(wire_store, vector_ids, lexical_ids, expected):
    wire_store(FakeStore([_row(cid) for cid in vector_ids], [_row(cid) for cid in lexical_ids]))

    result = _retrieve(search_mode="hybrid", top_k=10, rrf_k=10)

    assert [chunk.id for chunk in result.chunks] == expected
    assert result.low_confidence is (not expected)


def test_hybrid_applies_filters_to_both_branches_and_limits_after_fusion(wire_store):
    store = FakeStore([_row(1), _row(2)], [_row(3), _row(2)])
    wire_store(store)
    filters = {
        "sectors": ["finance"],
        "project_year_min": 2020,
        "project_year_max": 2025,
        "chunk_types": ["budget_component"],
    }

    result = _retrieve(search_mode="hybrid", top_k=1, recall_k=3, **filters)

    assert [chunk.id for chunk in result.chunks] == [2]
    for calls in (store.vector_calls, store.lexical_calls):
        assert all(calls[0][key] == value for key, value in filters.items())


def test_hybrid_reranking_receives_the_fused_candidates(wire_store):
    wire_store(FakeStore([_row(1), _row(2)], [_row(3), _row(2)]))
    seen = []

    class RecordingReranker:
        def rerank(self, query, candidates, *, top_n):
            seen.extend(chunk.id for chunk in candidates)
            return list(reversed(candidates))[:top_n]

    result = _retrieve(
        search_mode="hybrid", rerank=True, recall_k=3,
        rerank_top_n=2, reranker=RecordingReranker(),
    )

    assert seen == [2, 1, 3]
    assert [chunk.id for chunk in result.chunks] == [3, 1]


@pytest.mark.parametrize("kwargs", [{"search_mode": "invalid"}, {"rrf_k": 0}])
def test_retrieve_rejects_invalid_parameters_before_querying(wire_store, kwargs):
    store = FakeStore([], [])
    wire_store(store)

    with pytest.raises(ValueError):
        _retrieve(**kwargs)
    assert store.vector_calls == [] and store.lexical_calls == []


def test_rerank_reorders_and_cuts_to_top_n(wire_store):
    wire_store(FakeStore(vector_rows=[_row(1), _row(2), _row(3)], lexical_rows=[]))
    reranker = FakeReranker({"chunk-1": 0.1, "chunk-2": 0.5, "chunk-3": 0.9})

    result = _retrieve(search_mode="vector", rerank=True, rerank_top_n=2, reranker=reranker)

    assert [c.id for c in result.chunks] == [3, 2]


def test_retrieval_soft_fails_when_nothing_is_found(wire_store):
    wire_store(FakeStore(vector_rows=[], lexical_rows=[], candidates=7))

    result = _retrieve(search_mode="hybrid", rerank=False)

    assert result.chunks == []
    assert result.low_confidence is True
    assert result.candidates_evaluated == 7


@pytest.mark.parametrize("search_mode", ["vector", "hybrid"])
def test_cross_encoder_scores_top_50_and_keeps_top_5(wire_store, monkeypatch, search_mode):
    from app.rag.retrieval.reranker import CrossEncoderReranker

    store = FakeStore([_row(cid) for cid in range(1, 71)], [])
    wire_store(store)
    pairs_seen = []

    class Model:
        def predict(self, pairs):
            pairs_seen.extend(pairs)
            return [float(document.split("-")[1]) for _query, document in pairs]

    reranker = CrossEncoderReranker("test-cross-encoder")
    monkeypatch.setattr(reranker, "_ensure_loaded", lambda: Model())
    monkeypatch.setattr(deps, "get_reranker", lambda: reranker)

    result = _retrieve(search_mode=search_mode, rerank=True)

    assert store.vector_calls[0]["top_k"] == 50
    assert len(pairs_seen) == 50
    assert pairs_seen[0] == ("ecommerce checkout", "chunk-1")
    assert pairs_seen[-1] == ("ecommerce checkout", "chunk-50")
    assert [chunk.id for chunk in result.chunks] == [50, 49, 48, 47, 46]


@pytest.mark.parametrize("rerank,rows", [(False, [_row(1)]), (True, [])])
def test_reranker_is_not_resolved_when_disabled_or_candidates_are_empty(
    wire_store, monkeypatch, rerank, rows,
):
    wire_store(FakeStore(rows, []))

    def unexpected():
        pytest.fail("Disabled or empty retrieval must not resolve the cross-encoder")

    monkeypatch.setattr(deps, "get_reranker", unexpected)
    result = _retrieve(rerank=rerank)
    assert [chunk.id for chunk in result.chunks] == [row.id for row in rows]


def test_reranker_failure_is_a_retrieval_error_without_a_fallback(wire_store):
    from app.rag.errors import RetrievalError

    wire_store(FakeStore([_row(1)], []))

    class BrokenReranker:
        def rerank(self, *args, **kwargs):
            raise RuntimeError("Model inference failed")

    with pytest.raises(RetrievalError, match="Cross-encoder reranking failed") as exc:
        _retrieve(rerank=True, reranker=BrokenReranker())
    assert isinstance(exc.value.__cause__, RuntimeError)


# --- orchestrator ----------------------------------------------------------


class RecordingStore:
    def __init__(self):
        self.saved: dict[str, Estimate] = {}

    def get(self, key):
        return self.saved.get(key)

    def set(self, key, estimate):
        self.saved[key] = estimate


def _rchunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="Component: Checkout\nEstimated hours: 140",
        sector="ecommerce",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.42,
    )


def _good_estimate(*, cited=1) -> Estimate:
    return Estimate(
        total_engineer_days=18,
        duration_weeks=4,
        modules=[
            WorkModule(
                name="Checkout",
                tasks=[TaskItem(name="Cart & payment flow", engineer_days=18, sources=[cited])],
            )
        ],
        sources=[SourceCitation(source_id=cited, relevance="primary", used_for="checkout")],
        confidence="high",
        reasoning="Grounded in BUD-2024-005.",
    )


@pytest.fixture
def wire(monkeypatch):
    calls = {"reformulate": 0, "search": 0, "generate": 0, "embed": 0}
    store = RecordingStore()

    def _wire(*, retrieval: RetrievalResult, estimates: list[Estimate] | None = None):
        queue = list(estimates or [])

        async def fake_reformulate(transcript):
            calls["reformulate"] += 1
            return EstimationQuery(function="ecommerce storefront", sector="ecommerce")

        async def fake_retrieve(**kwargs):
            calls["search"] += 1
            calls["search_kwargs"] = kwargs
            return retrieval

        async def fake_generate(context_block, structured_query, **kwargs):
            calls["generate"] += 1
            return queue.pop(0)

        def fake_embed(text):
            calls["embed"] += 1
            return [0.0] * 1536

        runtime = SimpleNamespace(
            effective_search_mode=lambda: "hybrid", effective_rerank=lambda: True
        )
        monkeypatch.setattr(orch, "get_settings", lambda: _SETTINGS)
        monkeypatch.setattr(orch, "reformulate_query", fake_reformulate)
        monkeypatch.setattr(orch, "retrieve", fake_retrieve)
        monkeypatch.setattr(orch, "generate_estimate", fake_generate)
        monkeypatch.setattr(orch, "_generate", fake_generate)  # corrective retries
        monkeypatch.setattr(deps, "get_embedder", lambda: SimpleNamespace(embed_one=fake_embed))
        monkeypatch.setattr(deps, "get_token_encoder", lambda: CharEncoder())
        monkeypatch.setattr(deps, "get_idempotency_store", lambda: store)
        monkeypatch.setattr(deps, "get_runtime_retrieval_config", lambda: runtime)
        return calls, store

    return _wire


def _found(*ids: int) -> RetrievalResult:
    return RetrievalResult(
        chunks=[_rchunk(i) for i in ids], low_confidence=False, candidates_evaluated=5
    )


def test_happy_path_runs_every_stage_with_the_runtime_retrieval_mode(wire):
    calls, _ = wire(retrieval=_found(1), estimates=[_good_estimate()])

    result = asyncio.run(orch.estimate_from_transcript("x" * 200))

    assert (result.confidence, result.total_engineer_days) == ("high", 18)
    assert {k: calls[k] for k in ("reformulate", "search", "generate", "embed")} == {
        "reformulate": 1, "search": 1, "generate": 1, "embed": 1,
    }
    assert calls["search_kwargs"]["search_mode"] == "hybrid"
    assert calls["search_kwargs"]["rerank"] is True
    assert calls["search_kwargs"]["sectors"] == ["ecommerce"]


def test_soft_fail_skips_generation(wire):
    calls, _ = wire(
        retrieval=RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=7),
        estimates=[_good_estimate()],
    )

    result = asyncio.run(orch.estimate_from_transcript("x" * 200))

    assert result.confidence == "insufficient"
    assert result.total_engineer_days is None
    assert result.insufficient_context_explanation
    assert calls["generate"] == 0


def test_fabricated_citation_gets_one_corrective_retry(wire):
    calls, _ = wire(retrieval=_found(1), estimates=[_good_estimate(cited=99), _good_estimate()])

    result = asyncio.run(orch.estimate_from_transcript("x" * 200))

    assert result.confidence == "high"
    assert calls["generate"] == 2


def test_unrepaired_citations_downgrade_confidence_to_low(wire):
    calls, _ = wire(
        retrieval=_found(1), estimates=[_good_estimate(cited=99), _good_estimate(cited=98)]
    )

    result = asyncio.run(orch.estimate_from_transcript("x" * 200))

    assert result.confidence == "low"
    assert calls["generate"] == 2


def test_incoherent_insufficient_estimate_is_rejected_after_one_repair(wire):
    bad = Estimate(
        confidence="insufficient",
        reasoning="r",
        total_engineer_days=40,
        insufficient_context_explanation="should not carry numbers",
    )
    calls, _ = wire(retrieval=_found(1), estimates=[bad, bad])

    with pytest.raises(MalformedEstimateError):
        asyncio.run(orch.estimate_from_transcript("x" * 200))
    assert calls["generate"] == 2


def test_idempotent_replay_short_circuits_the_pipeline(wire):
    calls, store = wire(retrieval=_found(1), estimates=[_good_estimate()])

    first = asyncio.run(orch.estimate_from_transcript("x" * 200, idempotency_key="k1"))
    assert store.saved["k1"] == first
    second = asyncio.run(orch.estimate_from_transcript("x" * 200, idempotency_key="k1"))

    assert second == first
    assert {k: calls[k] for k in ("reformulate", "search", "generate", "embed")} == {
        "reformulate": 1, "search": 1, "generate": 1, "embed": 1,
    }


def test_generate_estimate_passes_reasoning_effort_and_token_ceiling(monkeypatch):
    captured: dict = {}

    def fake_complete_structured(**kwargs):
        captured.update(kwargs)
        return _good_estimate(), {}

    monkeypatch.setattr(orch, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(
        deps, "get_llm_wrapper", lambda: SimpleNamespace(complete_structured=fake_complete_structured)
    )

    estimate = asyncio.run(
        orch.generate_estimate('<source id="1">x</source>', EstimationQuery(function="store"))
    )

    assert estimate.confidence == "high"
    assert captured["max_tokens"] == 64_000
    assert captured["reasoning_effort"] == "high"
    assert captured["model_override"] == "gpt-5"
    assert captured["response_model"] is Estimate
