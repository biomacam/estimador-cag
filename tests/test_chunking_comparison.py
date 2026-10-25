import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.comparison import ChunkingComparator
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Budget
from app.embedding_pipeline.similarity import cosine_similarity, percentile
from app.embedding_pipeline.strategies import HierarchicalChunker
from app.main import app


@pytest.fixture
def budgets():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    return [Budget.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _keyword_embedder() -> OpenAIEmbedder:
    """Fake embeddings: 2-dim vector counting 'OAuth' vs everything else."""
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[float(t.count("OAuth")), 1.0]) for t in kwargs["input"]
        ]
    )
    return OpenAIEmbedder(client)


def test_cosine_similarity_and_percentile():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_run_queries_ranks_by_cosine(budgets):
    comparator = ChunkingComparator([JSONStructuralChunker()], _keyword_embedder())
    results = comparator.run_queries(budgets, ["OAuth OAuth"], top_k=2)
    top = results["structural"][0].top_k
    assert len(top) == 2
    assert top[0].cosine >= top[1].cosine
    assert "OAuth" in top[0].text_preview


def test_run_queries_without_queries_returns_empty(budgets):
    comparator = ChunkingComparator([JSONStructuralChunker()], Mock())
    assert comparator.run_queries(budgets, []) == {}


def test_chunking_is_memoized_between_stats_and_queries(budgets):
    chunker = JSONStructuralChunker()
    chunker.chunk = Mock(wraps=chunker.chunk)
    comparator = ChunkingComparator([chunker], _keyword_embedder())
    comparator.compute_stats(budgets)
    comparator.run_queries(budgets, ["OAuth"])
    chunker.chunk.assert_called_once()


def test_compute_stats_reports_distribution(budgets):
    stats = ChunkingComparator(
        [JSONStructuralChunker(), HierarchicalChunker()], Mock()
    ).compute_stats(budgets)
    n_components = sum(len(b.components) for b in budgets)
    assert stats["structural"].n_chunks == n_components
    assert stats["hierarchical"].n_chunks == n_components + len(budgets)
    td = stats["structural"].token_distribution
    assert td.min <= td.p50 <= td.p95 <= td.max


def test_build_chunkers_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        build_chunkers(["nope"])
    assert ALL_STRATEGIES[0] == "structural"


def test_compare_endpoint_returns_stats_and_top_k(client, budgets):
    app.dependency_overrides[get_embedder] = _keyword_embedder
    try:
        response = client.post(
            "/embeddings/compare",
            json={
                "budgets": [b.model_dump() for b in budgets],
                "queries": ["OAuth"],
                "strategies": ["structural", "hierarchical"],
                "top_k": 2,
            },
        )
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == 200
    body = response.json()
    assert set(body["stats_per_strategy"]) == {"structural", "hierarchical"}
    assert len(body["queries_per_strategy"]["structural"][0]["top_k"]) == 2


@pytest.mark.parametrize(
    "embedder,strategies,status",
    [(None, ["structural"], 500), ("ok", ["nope"], 400)],
)
def test_compare_endpoint_errors(client, budgets, embedder, strategies, status):
    app.dependency_overrides[get_embedder] = lambda: (
        _keyword_embedder() if embedder else None
    )
    try:
        response = client.post(
            "/embeddings/compare",
            json={"budgets": [budgets[0].model_dump()], "strategies": strategies},
        )
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert response.status_code == status
