"""CLI runner: feed the golden dataset against the estimator API.

Default transport is the in-process ``TestClient`` (no external server needed);
``--http BASE`` targets a running server instead. Either way the real LLM
providers configured in ``.env`` are called.

Usage::

    uv run python -m evals.run --mode actor
    uv run python -m evals.run --mode acb --limit 5 --output acb.json
    uv run python -m evals.run --mode actor --http http://localhost:8000

Prints one row per case with each metric's verdict and a final tally.
``--output`` persists the full report as JSON for diffing across runs. Exit
code is 0 only when every case passes every metric.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import httpx

from app.schemas.estimation import OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase, load_dataset
from evals.metrics import MetricResult, run_all_metrics

Mode = Literal["actor", "acb"]

_GUARDRAIL_OUT_OF_SCOPE_REASONS = {"prompt_injection", "moderation", "pii"}


def _form_body(case: GoldenCase) -> dict[str, str]:
    body = {
        "transcript": case.transcript,
        "project_type": case.project_type,
        "detail_level": case.detail_level,
        "output_format": case.output_format,
    }
    if case.tier != "auto":
        body["tier"] = case.tier
    return body


def _guardrail_payload(reason: str, message: str) -> dict[str, Any]:
    """Out-of-scope envelope for a layer-1 rejection: blocking an adversarial input
    before the LLM *is* the expected outcome, so it is scored like the LLM's own."""
    return {
        "result": {
            "summary": f"{OUT_OF_SCOPE_PREFIX} rejected by input guardrail ({reason}): {message}"[:1200],
            "confidence_pct": 0,
            "phases": [{
                "name": "n/a",
                "duration_weeks": 1,
                "cost_eur": 0,
                "summary": "Blocked by input guardrail; no estimation produced.",
            }],
            "total_duration_weeks": 1,
            "total_cost_eur": 0,
        },
        "prompt_version": f"guardrail/{reason}",
        "cached": False,
    }


def _maybe_guardrail_payload(case: GoldenCase, response: httpx.Response) -> dict[str, Any] | None:
    """Synthetic payload when a 400 guardrail rejection is this case's expected outcome."""
    if response.status_code != 400 or not case.expected_out_of_scope:
        return None
    try:
        detail = response.json().get("detail") or {}
    except ValueError:
        return None
    reason = detail.get("reason") if isinstance(detail, dict) else None
    if reason not in _GUARDRAIL_OUT_OF_SCOPE_REASONS:
        return None
    return _guardrail_payload(reason, detail.get("message", ""))


def run_case(client: httpx.Client, case: GoldenCase, mode: Mode) -> tuple[dict[str, Any], int]:
    """One fresh session per case so cases never contaminate each other."""
    session_id = client.post("/sessions").json()["session_id"]
    endpoint = f"/sessions/{session_id}/estimate-acb" if mode == "acb" else f"/sessions/{session_id}/estimate"
    t0 = time.perf_counter()
    response = client.post(endpoint, data=_form_body(case))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    synthetic = _maybe_guardrail_payload(case, response)
    if synthetic is not None:
        return synthetic, latency_ms
    response.raise_for_status()
    return response.json(), latency_ms


def evaluate(case: GoldenCase, payload: dict[str, Any]) -> list[MetricResult]:
    return run_all_metrics(case, EstimationResult.model_validate(payload["result"]))


def run_cases(client: httpx.Client, cases: list[GoldenCase], mode: Mode) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_totals: dict[str, list[bool]] = {}
    pass_count = 0

    print(f"Running {len(cases)} cases against mode={mode}")
    for case in cases:
        try:
            payload, latency_ms = run_case(client, case, mode)
        except Exception as exc:  # noqa: BLE001 — one broken case must not abort the run
            print(f"{case.id}: ERROR {type(exc).__name__}: {str(exc)[:200]}")
            rows.append({"id": case.id, "error": str(exc)[:400]})
            continue

        metrics = evaluate(case, payload)
        print(_format_row(case.id, latency_ms, metrics))
        passed = all(m.passed for m in metrics)
        pass_count += passed
        rows.append({
            "id": case.id,
            "latency_ms": latency_ms,
            "passed": passed,
            "metrics": [m.__dict__ for m in metrics],
            "result_summary": payload["result"]["summary"][:200],
            "prompt_version": payload.get("prompt_version"),
            "acb_final_decision": payload.get("acb", {}).get("final_decision"),
        })
        for m in metrics:
            metric_totals.setdefault(m.name, []).append(m.passed)

    print("\nSummary")
    print(f"  {pass_count}/{len(cases)} cases passing all metrics")
    for name, results in metric_totals.items():
        print(f"  {name}: {sum(results)}/{len(results)}")

    return {"mode": mode, "cases_total": len(cases), "cases_passing": pass_count, "rows": rows}


def _format_row(case_id: str, latency_ms: int, metrics: list[MetricResult]) -> str:
    parts = [f"{case_id:<34}", f"{latency_ms:>6} ms"]
    parts.extend(f"{m.name}={'PASS' if m.passed else 'FAIL'}({m.score:.2f})" for m in metrics)
    return " | ".join(parts)


@contextmanager
def _in_process_client() -> Iterator[httpx.Client]:
    """``TestClient`` over the app with an isolated session store for the whole run."""
    from fastapi.testclient import TestClient

    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions import SessionStore

    # One store for the run: a per-request store would 404 the turn after POST /sessions.
    eval_store = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: eval_store
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_session_store, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("actor", "acb"), default="actor")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--http", default=None, help="Base URL of a running server (e.g. http://localhost:8000)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = load_dataset()[: args.limit]

    if args.http:
        with httpx.Client(base_url=args.http, timeout=180.0) as client:
            report = run_cases(client, cases, args.mode)
    else:
        with _in_process_client() as client:
            report = run_cases(client, cases, args.mode)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    return 0 if report["cases_passing"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
