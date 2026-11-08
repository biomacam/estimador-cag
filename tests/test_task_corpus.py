"""The task-corpus generator output is valid, ingestable and deterministic, and the
``module`` / ``chunk_type`` plumbing it relies on reaches the chunks."""

from __future__ import annotations

import sys
from pathlib import Path

from app.embedding_pipeline.chunker import JSONStructuralChunker, render_component_text
from app.embedding_pipeline.schemas import Budget, IngestRequest

# The generator lives in scripts/, which is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_task_corpus import CHUNK_TYPE, DOCUMENT_TYPE, generate_corpus  # noqa: E402


def test_generated_corpus_validates_as_budget():
    budgets = [Budget.model_validate(p) for p in generate_corpus(count=6, seed=1)]

    assert len(budgets) == 6
    for budget in budgets:
        assert budget.components, "each project has at least one task"
        assert all(c.module for c in budget.components)
        assert budget.total_estimated_hours == sum(c.estimated_hours for c in budget.components)


def test_generation_is_deterministic_for_a_seed():
    assert generate_corpus(count=4, seed=7) == generate_corpus(count=4, seed=7)


def test_budget_ids_are_unique():
    ids = [p["budget_id"] for p in generate_corpus(count=12, seed=90)]
    assert len(ids) == len(set(ids))


def test_corpus_is_task_granular():
    corpus = generate_corpus(count=12, seed=90)

    assert sum(len(p["components"]) for p in corpus) >= 120
    assert len({c["module"] for p in corpus for c in p["components"]}) >= 8


def test_committed_corpus_matches_the_generator_defaults():
    import json

    committed = json.loads((Path(__file__).resolve().parents[1] / "data" / "task_corpus.json").read_text(encoding="utf-8"))
    assert committed == generate_corpus()


def test_module_reaches_chunk_text_and_metadata():
    budget = Budget.model_validate(generate_corpus(count=1, seed=3)[0])
    component = budget.components[0]

    chunk = JSONStructuralChunker().chunk([budget])[0]

    assert f"Module: {component.module}\n" in chunk.text
    assert chunk.metadata["module"] == component.module


def test_component_without_module_keeps_the_previous_text():
    budget = Budget.model_validate(generate_corpus(count=1, seed=3)[0])
    component = budget.components[0].model_copy(update={"module": None})

    assert "Module:" not in render_component_text(budget, component)


def test_ingest_request_chunk_type_defaults_to_budget_component():
    content = generate_corpus(count=1, seed=3)[0]
    base = {"source_path": "x", "document_type": DOCUMENT_TYPE, "content": content}

    assert IngestRequest.model_validate(base).chunk_type == "budget_component"
    assert IngestRequest.model_validate({**base, "chunk_type": CHUNK_TYPE}).chunk_type == CHUNK_TYPE
