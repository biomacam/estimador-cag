"""The retrieval-evaluation harness: golden set integrity and the precision metric."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_retrieval_s10 import CONFIGS, precision_at_k  # noqa: E402

from app.config import Settings  # noqa: E402


def _golden() -> dict:
    return json.loads((ROOT / "evals" / "golden_retrieval.json").read_text(encoding="utf-8"))


def test_golden_relevant_budgets_exist_in_the_ingested_corpus():
    corpus_ids = {b["budget_id"] for b in json.loads((ROOT / "data" / "budgets_sample.json").read_text(encoding="utf-8"))}
    golden = _golden()

    assert golden["chunk_types"] == ["budget_component"]
    for query in golden["queries"]:
        assert query["relevant_budget_ids"], query["id"]
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
