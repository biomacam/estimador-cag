"""Three deterministic metrics for the multi-turn / large-attachment stress runs.

They live apart from ``evals.metrics`` because they take a ``TurnObservation`` (the
telemetry of ``turn_observed``) or a session snapshot dict (the JSON of
``GET /sessions/{id}``), not a ``(GoldenCase, EstimationResult)`` pair. ``MetricResult``
is reused so reports keep one shape.

No embeddings and no LLM judge: ``MemoryDriftMetric`` is a case-insensitive substring
match, so when it says a fact is gone, the fact is gone.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from app.schemas.estimation import TurnObservation
from evals.metrics import MetricResult

FactField = Literal["project_name", "technologies", "scope", "summary", "any"]


class LatencyBudgetMetric:
    """1.0 if ``latency_ms`` <= ``budget_ms``, else 0.0."""

    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms must be positive")
        self.budget_ms = budget_ms

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        passed = observation.latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"{observation.latency_ms} ms vs budget {self.budget_ms} ms",
        )


class CostBudgetMetric:
    """1.0 if the turn's ``cost_usd`` <= ``budget_usd``, else 0.0 (per turn, not cumulative)."""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = budget_usd

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        passed = observation.cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"${observation.cost_usd:.6f} vs budget ${self.budget_usd:.6f}",
        )


class MemoryDriftMetric:
    """1.0 if a fact from an earlier turn still appears in a later session snapshot.

    ``fact_field`` scopes the search (the same values the scenarios declare):

    - ``project_name`` / ``scope``: ``metadata.project_name`` / ``metadata.agreed_scope``
    - ``technologies``: ``metadata.mentioned_technologies``
    - ``summary``: the cumulative summary
    - ``any``: metadata, summary and anchors together (never counters such as ``message_count``)
    """

    name = "memory_drift"

    def __init__(self, fact: str, fact_field: FactField = "any") -> None:
        if not fact:
            raise ValueError("fact must be a non-empty string")
        self.fact = fact
        self.fact_field = fact_field

    def evaluate(self, snapshot: dict[str, Any]) -> MetricResult:
        found = self.fact.lower() in self._haystack(snapshot).lower()
        return MetricResult(
            name=self.name,
            score=1.0 if found else 0.0,
            passed=found,
            details=f"fact={self.fact!r} field={self.fact_field} {'present' if found else 'missing'}",
        )

    def _haystack(self, snapshot: dict[str, Any]) -> str:
        metadata = snapshot.get("metadata") or {}
        if self.fact_field == "project_name":
            return str(metadata.get("project_name") or "")
        if self.fact_field == "technologies":
            return " ".join(metadata.get("mentioned_technologies") or [])
        if self.fact_field == "scope":
            return str(metadata.get("agreed_scope") or "")
        if self.fact_field == "summary":
            return str(snapshot.get("summary") or "")
        memory = {key: snapshot.get(key) for key in ("metadata", "summary", "anchors")}
        return json.dumps(memory, default=str)
