"""Unit tests for the agent tools (no network, no DB)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agentic import agent_tools
from app.agentic.agent_schemas import SearchBudgetsArgs
from app.agentic.agent_tools import (
    CONTINGENCY_FACTOR,
    calculate_estimate,
    default_retrieval_backend,
    dispatch_tool,
    search_budgets,
    validate_estimate,
)
from app.rag.schemas import RetrievalResult, RetrievedChunk


# --- calculate_estimate ---------------------------------------------------- #
def test_calculate_estimate_median_plus_contingency():
    result = calculate_estimate(
        {
            "components": [
                {"name": "Auth backend", "reference_amounts": [100.0, 200.0, 300.0]}
            ]
        }
    )
    # median(100,200,300)=200; +15% contingency => 230.
    assert result["components"][0]["estimated_hours"] == pytest.approx(
        200 * (1 + CONTINGENCY_FACTOR)
    )
    assert result["total_hours"] == pytest.approx(230.0)
    assert result["components"][0]["unbudgeted"] is False


def test_calculate_estimate_flags_unbudgeted_without_inventing_hours():
    result = calculate_estimate(
        {"components": [{"name": "Mystery module", "reference_amounts": []}]}
    )
    component = result["components"][0]
    assert component["estimated_hours"] == 0.0
    assert component["unbudgeted"] is True
    assert result["total_hours"] == 0.0


def test_calculate_estimate_sums_components():
    result = calculate_estimate(
        {
            "components": [
                {"name": "A", "reference_amounts": [100.0]},
                {"name": "B", "reference_amounts": [200.0]},
            ]
        }
    )
    assert result["total_hours"] == pytest.approx(115.0 + 230.0)


def test_calculate_estimate_rejects_bad_args():
    with pytest.raises(ValidationError):
        calculate_estimate({"components": [{"name": "A"}]})  # missing reference_amounts


# --- validate_estimate ----------------------------------------------------- #
def test_validate_estimate_passes_clean_estimate():
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 115.0, "reference_amounts": [100.0]}
            ],
            "total_hours": 115.0,
        }
    )
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_estimate_flags_unbudgeted_and_total_mismatch():
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 50.0, "reference_amounts": []}
            ],
            "total_hours": 999.0,
        }
    )
    assert result["ok"] is False
    joined = " ".join(result["issues"]).lower()
    assert "no historical reference" in joined
    assert "does not match" in joined


def test_validate_estimate_flags_out_of_range_component():
    # reference 100 -> plausible range [50, 200]; 1000 is out of range.
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 1000.0, "reference_amounts": [100.0]}
            ],
            "total_hours": 1000.0,
        }
    )
    assert result["ok"] is False
    assert any("outside the plausible range" in issue for issue in result["issues"])


def test_validate_estimate_flags_nonpositive_total():
    result = validate_estimate({"components": [], "total_hours": 0.0})
    assert result["ok"] is False
    assert any("non-positive" in issue for issue in result["issues"])


# --- search_budgets + dispatch --------------------------------------------- #
def test_search_budgets_uses_injected_backend():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        assert args.query == "auth backend"
        return [
            {"id": 1, "estimated_hours": 420.0, "content_preview": "x", "distance": 0.1}
        ]

    result = asyncio.run(
        search_budgets({"query": "auth backend", "filters": None}, backend=fake_backend)
    )
    assert result["count"] == 1
    assert "420.0" in result["summary"]


def test_search_budgets_rejects_empty_query():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValidationError):
        asyncio.run(
            search_budgets({"query": "", "filters": None}, backend=fake_backend)
        )


def test_dispatch_unknown_tool_raises():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValueError, match="Unknown tool"):
        asyncio.run(dispatch_tool("nonexistent", {}, backend=fake_backend))


# --- default_retrieval_backend --------------------------------------------- #
def test_default_backend_searches_historical_tasks_and_adapts_chunks(monkeypatch):
    calls: list[dict] = []

    async def fake_retrieve(**kwargs):
        calls.append(kwargs)
        chunk = RetrievedChunk(
            id=7,
            content="Task:   SAP\n  sync   " + "x" * 300,
            sector="logistics",
            project_year=2024,
            chunk_type="historical_task",
            distance=0.123456,
            budget_id="BUD-7",
            estimated_hours=120,
        )
        return RetrievalResult(
            chunks=[chunk], low_confidence=False, candidates_evaluated=1
        )

    embedder = SimpleNamespace(embed_one=lambda text: [0.1, 0.2])
    monkeypatch.setattr(agent_tools, "get_embedder", lambda: embedder)
    monkeypatch.setattr(agent_tools, "retrieve", fake_retrieve)

    items = asyncio.run(
        default_retrieval_backend(
            SearchBudgetsArgs(
                query="SAP integration", filters={"sectors": ["logistics"]}
            )
        )
    )

    assert calls[0]["chunk_types"] == ["budget_component", "historical_task"]
    assert calls[0]["sectors"] == ["logistics"]
    assert calls[0]["query_embedding"] == [0.1, 0.2]
    item = items[0]
    assert item["estimated_hours"] == 120
    assert item["chunk_type"] == "historical_task"
    assert item["distance"] == 0.1235
    assert item["content_preview"].startswith("Task: SAP sync x")
    assert len(item["content_preview"]) == agent_tools.CONTENT_PREVIEW_CHARS


def test_default_backend_requires_embedder(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_embedder", lambda: None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(default_retrieval_backend(SearchBudgetsArgs(query="x")))
