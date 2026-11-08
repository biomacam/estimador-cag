"""The retrieval-evaluation harness: golden set integrity and the precision metric."""

from __future__ import annotations

import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_retrieval_s10 as evaluation  # noqa: E402
from eval_retrieval_s10 import CONFIGS, format_report, precision_at_k  # noqa: E402

from app.config import Settings  # noqa: E402


def _golden() -> dict:
    return json.loads((ROOT / "evals" / "golden_retrieval.json").read_text(encoding="utf-8"))


def test_golden_relevant_budgets_exist_in_the_ingested_corpus():
    corpus_ids = {b["budget_id"] for b in json.loads((ROOT / "data" / "budgets_sample.json").read_text(encoding="utf-8"))}
    golden = _golden()

    assert len(golden["queries"]) == 5
    assert len({query["id"] for query in golden["queries"]}) == 5
    assert golden["k"] == 5
    assert golden["chunk_types"] == ["budget_component"]
    for query in golden["queries"]:
        assert query["relevant_budget_ids"], query["id"]
        assert query["note"], query["id"]
        assert set(query["relevant_budget_ids"]) <= corpus_ids, query["id"]


def test_the_four_configurations_cover_every_search_and_rerank_combination():
    assert {(mode, rerank) for _id, _s, _r, mode, rerank in CONFIGS} == {
        ("vector", False), ("hybrid", False), ("vector", True), ("hybrid", True),
    }


def test_precision_at_k_counts_relevant_budgets_in_the_top_k_only():
    chunks = [SimpleNamespace(budget_id=b) for b in ("A", "X", "B", "Y", "Z", "A")]

    assert precision_at_k(chunks, {"A", "B"}, 5) == 2 / 5  # the 6th chunk is ignored
    assert precision_at_k(chunks, set(), 5) == 0.0


def test_reranking_is_enabled_by_default():
    assert Settings.model_fields["RERANKER_ENABLED"].default is True


def test_precision_uses_fixed_denominator_for_short_rankings():
    assert precision_at_k([SimpleNamespace(budget_id="A")], {"A"}, 5) == 0.2
    with pytest.raises(ValueError, match="must be positive"):
        precision_at_k([], set(), 0)


def test_report_contains_comparison_annotations_and_lexical_diagnostics():
    results = {
        cfg_id: {
            "precisions": [0.6], "latencies_ms": [10.0],
            "per_query": {"Q1": {"precision_at_5": 0.6}},
        }
        for cfg_id, *_ in CONFIGS
    }
    queries = [{"id": "Q1", "query": "A project description", "relevant_budget_ids": ["BUD-1"]}]
    report = format_report(results, queries, 5, {"Q1": 0})

    assert "| A | Vector | No | 0.60 | 10.0 |" in report
    assert "| D | Hybrid | Yes | 0.60 | 10.0 |" in report
    assert "BUD-1" in report
    assert "No lexical matches were found" in report
    assert "excludes embeddings" in report


@pytest.mark.parametrize("invalid", ["missing", "extra", "unembedded"])
def test_evaluation_rejects_incomplete_or_contaminated_corpus(monkeypatch, invalid):
    from app.embedding_pipeline.chunker import JSONStructuralChunker
    from app.embedding_pipeline.schemas import Budget

    corpus = ROOT / "data" / "budgets_sample.json"
    chunks = JSONStructuralChunker().chunk([
        Budget.model_validate(budget) for budget in json.loads(corpus.read_text(encoding="utf-8"))
    ])
    rows = [(chunk.text, chunk.metadata, True) for chunk in chunks]
    if invalid == "missing":
        rows.pop()
    elif invalid == "extra":
        rows.append(("Unexpected content", {"budget_id": "unknown"}, True))
    else:
        rows[0] = (rows[0][0], rows[0][1], False)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, statement):
            return SimpleNamespace(all=lambda: rows)

    monkeypatch.setattr(evaluation, "get_async_session_factory", lambda: Session)
    with pytest.raises(ValueError, match="complete base corpus"):
        asyncio.run(evaluation.verify_corpus(corpus, ["budget_component"]))
