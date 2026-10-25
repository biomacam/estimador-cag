import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAI

from app.embedding_pipeline.base import Chunker, count_tokens
from app.embedding_pipeline.chunker import JSONStructuralChunker, serialize_budget
from app.embedding_pipeline.schemas import Budget
from app.embedding_pipeline.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.embedding_pipeline.strategies import fixed_size


@pytest.fixture
def budgets():
    path = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"
    return [Budget.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _n_components(budgets) -> int:
    return sum(len(b.components) for b in budgets)


@pytest.mark.parametrize(
    "chunker_cls,name",
    [
        (JSONStructuralChunker, "structural"),
        (FixedSizeChunker, "fixed_size"),
        (RecursiveChunker, "recursive"),
        (HierarchicalChunker, "hierarchical"),
        (SentenceWindowChunker, "sentence_window"),
    ],
)
def test_offline_strategies_share_interface(budgets, chunker_cls, name):
    chunker = chunker_cls()
    chunks = chunker.chunk(budgets)
    assert isinstance(chunker, Chunker)
    assert chunker.strategy_name == name
    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.token_count == count_tokens(c.text) for c in chunks)
    assert chunker.last_extra_api_calls == 0
    assert chunker.last_extra_cost_usd == 0.0
    assert chunker.chunk([]) == []


def test_fixed_size_splits_long_component_with_overlap(budgets, monkeypatch):
    monkeypatch.setattr(fixed_size, "CHUNK_SIZE_TOKENS", 20)
    monkeypatch.setattr(fixed_size, "OVERLAP_TOKENS", 5)
    chunks = FixedSizeChunker().chunk(budgets[:1])
    first_id = f"{budgets[0].budget_id}::{budgets[0].components[0].component_id}"
    parts = [c for c in chunks if c.chunk_id.startswith(f"{first_id}::p")]
    assert len(parts) > 1
    assert all(c.token_count <= 20 for c in parts)
    assert [c.metadata["part"] for c in parts] == list(range(len(parts)))


def test_recursive_respects_token_budget(budgets):
    for chunk in RecursiveChunker().chunk(budgets):
        assert chunk.token_count <= 512
        assert "component_id" not in chunk.metadata


def test_hierarchical_links_children_to_parent(budgets):
    chunks = HierarchicalChunker().chunk(budgets)
    parents = {c.chunk_id: c for c in chunks if c.metadata["level"] == "parent"}
    children = [c for c in chunks if c.metadata["level"] == "child"]
    assert len(parents) == len(budgets)
    assert len(children) == _n_components(budgets)
    for budget in budgets:
        assert parents[f"{budget.budget_id}::parent"].text == serialize_budget(budget)
    assert all(c.metadata["parent_chunk_id"] in parents for c in children)


def test_sentence_window_embeds_sentence_and_keeps_window(budgets):
    for chunk in SentenceWindowChunker().chunk(budgets):
        assert chunk.text in chunk.metadata["window_text"]


def test_semantic_requires_api_key():
    with pytest.raises(RuntimeError):
        SemanticChunker(api_key=None)


def test_semantic_reports_extra_embedding_cost(budgets):
    chunker = SemanticChunker(api_key="sk-test")
    chunker._splitter = Mock(split_text=lambda text: [text[:50], text[50:]])
    chunks = chunker.chunk(budgets)
    assert len(chunks) == 2 * len(budgets)
    assert chunker.last_extra_api_calls == len(budgets)
    assert chunker.last_extra_cost_usd > 0


def test_propositional_splits_and_falls_back_on_error(budgets, monkeypatch):
    chunker = PropositionalChunker(client=OpenAI(api_key="sk-test"))
    results = iter(
        [(["Fact A.", "Fact B."], 0.001)] + [([], 0.0)] * (_n_components(budgets) - 1)
    )
    monkeypatch.setattr(chunker, "_decompose", lambda text: next(results))
    chunks = chunker.chunk(budgets)
    assert [c.text for c in chunks[:2]] == ["Fact A.", "Fact B."]
    assert len(chunks) == 2 + _n_components(budgets) - 1
    assert chunker.last_extra_api_calls == _n_components(budgets)
    assert chunker.last_extra_cost_usd == pytest.approx(0.001)


def test_contextual_retrieval_prepends_generated_context(budgets):
    client = Mock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="  Situating context.  ")],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=1000,
        ),
    )
    chunker = ContextualRetrievalChunker(client=client)
    chunks = chunker.chunk(budgets)
    assert len(chunks) == _n_components(budgets)
    assert all(c.text.startswith("Situating context.\n\n[Project:") for c in chunks)
    assert all(c.metadata["generated_context"] == "Situating context." for c in chunks)
    assert chunker.last_extra_api_calls == _n_components(budgets)
    assert chunker.last_extra_cost_usd > 0
    system = client.messages.create.call_args.kwargs["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"}


def test_contextual_retrieval_degrades_to_base_text_on_error(budgets):
    client = Mock()
    client.messages.create.side_effect = RuntimeError("boom")
    chunks = ContextualRetrievalChunker(client=client).chunk(budgets[:1])
    assert all(c.text.startswith("[Project:") for c in chunks)
    assert all(c.metadata["generated_context"] == "" for c in chunks)
