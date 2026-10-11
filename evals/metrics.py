"""Deterministic metrics over a ``(GoldenCase, EstimationResult)`` pair.

Kept in-tree (no LLM judge) so the suite says exactly *why* a case failed and
scoring itself never needs a provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


def _result(name: str, problems: list[str], ok: str) -> MetricResult:
    return MetricResult(
        name=name,
        score=0.0 if problems else 1.0,
        passed=not problems,
        details="; ".join(problems) or ok,
    )


class SchemaAdherenceMetric:
    """Flags regressions of the guarantees Instructor + validators already enforce,
    and checks the phase count against the expected range when provided."""

    name = "schema_adherence"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        phase_sum = sum(p.cost_eur for p in result.phases)
        if phase_sum != result.total_cost_eur:
            problems.append(f"phases sum ({phase_sum}) != total_cost_eur ({result.total_cost_eur})")

        if result.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not result.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            problems.append(f"low confidence missing '{OUT_OF_SCOPE_PREFIX}' prefix")

        if case.expected_phase_count_range:
            lo, hi = case.expected_phase_count_range
            if not lo <= len(result.phases) <= hi:
                problems.append(f"phase count {len(result.phases)} outside [{lo}, {hi}]")

        return _result(self.name, problems, "schema ok")


class CostBoundsMetric:
    """Sanity bounds for cost and duration. Out-of-scope cases expect the
    placeholder envelope (cost=0, duration=1)."""

    name = "cost_bounds"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        if case.expected_out_of_scope:
            if result.total_cost_eur != 0 or result.total_duration_weeks != 1:
                problems.append(
                    "expected out-of-scope envelope (cost=0, duration=1), got "
                    f"cost={result.total_cost_eur}, weeks={result.total_duration_weeks}"
                )
        else:
            if case.expected_cost_range_eur:
                lo, hi = case.expected_cost_range_eur
                if not lo <= result.total_cost_eur <= hi:
                    problems.append(f"cost {result.total_cost_eur} EUR outside [{lo}, {hi}]")
            if case.expected_duration_weeks_range:
                lo, hi = case.expected_duration_weeks_range
                if not lo <= result.total_duration_weeks <= hi:
                    problems.append(f"duration {result.total_duration_weeks}w outside [{lo}, {hi}]")

        return _result(self.name, problems, "cost & duration within bounds")


class ContentRecallMetric:
    """Does the summary / phase prose mention what the user cared about?
    Out-of-scope cases skip this."""

    name = "content_recall"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        if case.expected_out_of_scope:
            return MetricResult(self.name, 1.0, True, "skipped — case is out-of-scope by design")

        haystack = " ".join(
            [result.summary] + [f"{phase.summary} {phase.name}" for phase in result.phases]
        ).lower()

        problems: list[str] = []
        missing = [term for term in case.expected_in_summary if term.lower() not in haystack]
        if missing:
            problems.append(f"missing in summary: {missing}")
        if case.expected_technologies_any_of and not any(
            tech.lower() in haystack for tech in case.expected_technologies_any_of
        ):
            problems.append(f"none of {case.expected_technologies_any_of} mentioned anywhere")

        return _result(self.name, problems, "expected content present")


_DEFAULT_METRICS = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())


def run_all_metrics(case: GoldenCase, result: EstimationResult) -> list[MetricResult]:
    return [metric.evaluate(case, result) for metric in _DEFAULT_METRICS]
