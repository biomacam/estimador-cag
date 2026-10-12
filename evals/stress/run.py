"""Stress runner: scenarios x attachment sizes x repeats against a running estimator.

Per turn it POSTs ``/sessions/{id}/estimate`` (with the pre-built PDF, if any), reads
``response.observation`` (the unified ``turn_observed`` telemetry, so no log parsing),
GETs ``/sessions/{id}`` for the post-turn snapshot, scores the three metrics and writes
one CSV row.

Memory is scored two ways on the snapshot (summary text, anchors and metadata):
``memory_drift_passed`` tracks the turn-1 fact, ``fact_recall`` is the share of *all*
facts declared in earlier turns that are still present. Scenarios are prefixes of one
20-turn script, so a single run covers N in {1, 3, 6, 10, 20}.

Default transport is the in-process ``TestClient``; ``--http BASE`` targets a running
server. Either way the real LLM providers configured in ``.env`` are called.

    uv run python -m evals.stress.fixtures.build_pdfs
    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 --output evals/stress/results.csv
    uv run python -m evals.stress.report evals/stress/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from app.schemas.estimation import TurnObservation
from evals.stress.fixtures.build_pdfs import pdf_path
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import Scenario, ScenarioTurn, get_scenario

MAX_TURN_RETRIES = 1

CSV_COLUMNS = [
    "scenario",
    "attachment_size_kb",
    "repeat",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "wall_clock_ms",
    "turn_retries",
    "cache_hit_kind",
    "last_resolved_tier",
    "latency_budget_passed",
    "cost_budget_passed",
    "memory_drift_passed",
    "tracked_fact",
    "fact_recall",
    "facts_checked",
    "error",
]


@contextmanager
def _open_client(http_base_url: str | None) -> Iterator[httpx.Client | Any]:
    if http_base_url:
        with httpx.Client(base_url=http_base_url, timeout=180.0) as client:
            yield client
        return

    from fastapi.testclient import TestClient

    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions import SessionStore

    # One store for the run: a per-request store would 404 the turn after POST /sessions.
    store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session_store, None)


def _form_body(scenario: Scenario, turn: ScenarioTurn) -> dict[str, str]:
    return {
        "transcript": turn.transcript,
        "project_type": scenario.project_type.value,
        "detail_level": scenario.detail_level.value,
        "output_format": scenario.output_format.value,
    }


def _attachment_files(size_kb: int) -> list[tuple[str, tuple[str, bytes, str]]] | None:
    if size_kb == 0:
        return None
    path = pdf_path(size_kb)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} missing; run `uv run python -m evals.stress.fixtures.build_pdfs`."
        )
    return [("attachments", (path.name, path.read_bytes(), "application/pdf"))]


def _execute_turn(
    client: httpx.Client, session_id: str, scenario: Scenario, turn: ScenarioTurn, files: list | None
) -> tuple[TurnObservation, dict[str, Any], int, int]:
    """One POST estimate + one GET snapshot; returns ``(observation, snapshot, wall_clock_ms, retries)``.

    ``wall_clock_ms`` (last attempt) includes the round trip, attachment extraction, compression
    and the metadata pass; ``observation.latency_ms`` is only the main LLM call. A 502 means the
    server failed before committing the turn, so it is retried once; nothing else is, because a
    timeout may still commit server-side and a retry would duplicate the turn.
    """
    retries = 0
    while True:
        started = time.perf_counter()
        response = client.post(
            f"/sessions/{session_id}/estimate", data=_form_body(scenario, turn), files=files
        )
        wall_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code == 502 and retries < MAX_TURN_RETRIES:
            retries += 1
            continue
        response.raise_for_status()
        break

    observation = response.json().get("observation")
    if observation is None:
        raise RuntimeError("response.observation is missing; is this estimator built with turn_observed?")

    snapshot = client.get(f"/sessions/{session_id}")
    snapshot.raise_for_status()
    return TurnObservation(**observation), snapshot.json(), wall_ms, retries


def _describe_error(exc: Exception) -> str:
    """Short error text; for HTTP errors it carries the server's ``reason`` code and message."""
    # Duck-typed: the in-process TestClient raises httpx2's HTTPStatusError, not httpx's.
    response = getattr(exc, "response", None)
    if response is None or not hasattr(response, "status_code"):
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        body = response.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    reason = body.get("reason") or {}
    detail = body.get("detail")
    code = reason.get("code") or (detail.get("reason") if isinstance(detail, dict) else "")
    message = reason.get("message") or ""
    text = f"HTTP {response.status_code} {code or ''}".strip()
    return f"{text}: {message}"[:300] if message else text


def fact_recall(scenario: Scenario, turn_index: int, snapshot: dict[str, Any]) -> tuple[float | None, int]:
    """Share of facts introduced before ``turn_index`` that the snapshot still holds."""
    facts = {(t.fact_to_remember, t.fact_field) for t in scenario.facts_before(turn_index)}
    if not facts:
        return None, 0
    found = sum(MemoryDriftMetric(fact, field).evaluate(snapshot).passed for fact, field in facts)
    return found / len(facts), len(facts)


def _run_one_session(
    client: httpx.Client,
    scenario: Scenario,
    size_kb: int,
    repeat: int,
    latency_metric: LatencyBudgetMetric,
    cost_metric: CostBudgetMetric,
    writer: csv.DictWriter,
) -> int:
    """Walk the scenario once in a fresh session; returns the rows written."""
    files = _attachment_files(size_kb)
    create = client.post("/sessions")
    create.raise_for_status()
    session_id = create.json()["session_id"]

    first = scenario.turns[0]
    drift_metric = (
        MemoryDriftMetric(first.fact_to_remember, first.fact_field) if first.fact_to_remember else None
    )

    rows = 0
    for turn in scenario.turns:
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "attachment_size_kb": size_kb,
            "repeat": repeat,
            "turn_index": turn.turn_index,
            "session_id": session_id,
            "tracked_fact": first.fact_to_remember or "",
        }
        try:
            observation, snapshot, wall_ms, retries = _execute_turn(
                client, session_id, scenario, turn, files
            )
        except Exception as exc:  # noqa: BLE001 — a mid-conversation failure would skew later turns
            row["error"] = _describe_error(exc)
            writer.writerow(row)
            return rows + 1

        recall, checked = fact_recall(scenario, turn.turn_index, snapshot)
        row.update(
            {
                **observation.model_dump(),
                "turn_index": turn.turn_index,
                "wall_clock_ms": wall_ms,
                "turn_retries": retries,
                "latency_budget_passed": latency_metric.evaluate(observation).passed,
                "cost_budget_passed": cost_metric.evaluate(observation).passed,
                # Turn 1 trivially remembers itself.
                "memory_drift_passed": (
                    "" if drift_metric is None or turn.turn_index == 1
                    else drift_metric.evaluate(snapshot).passed
                ),
                "fact_recall": "" if recall is None else round(recall, 4),
                "facts_checked": checked,
                "error": "",
            }
        )
        writer.writerow(row)
        rows += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", default=None, help="Base URL of a running estimator; default is in-process")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100", help="PDF sizes in KB; 0 = none")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-budget-ms", type=int, default=8000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args(argv)

    scenarios = [get_scenario(name.strip()) for name in args.scenarios.split(",")]
    sizes = [int(size) for size in args.attachment_sizes.split(",")]
    for size in sizes:  # fail before a long run, not in the middle of it
        _attachment_files(size)

    latency_metric = LatencyBudgetMetric(args.latency_budget_ms)
    cost_metric = CostBudgetMetric(args.cost_budget_usd)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Stress run: {len(scenarios)} scenarios x {len(sizes)} sizes x {args.repeats} repeats")
    total = 0
    started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as fh, _open_client(args.http) as client:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for scenario in scenarios:
            for size_kb in sizes:
                for repeat in range(args.repeats):
                    print(f"  {scenario.name:<14} kb={size_kb:>3} repeat={repeat + 1}/{args.repeats}")
                    total += _run_one_session(
                        client, scenario, size_kb, repeat, latency_metric, cost_metric, writer
                    )
                    fh.flush()

    print(f"\nWrote {total} rows to {args.output} in {int(time.perf_counter() - started)}s")
    print(f"Next: uv run python -m evals.stress.report {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
