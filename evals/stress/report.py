"""Build ``REPORT.md`` tables from a stress ``results.csv`` (no LLM, fully deterministic).

    uv run python -m evals.stress.report evals/stress/results.csv [--output evals/stress/REPORT.md]

The numeric sections (summary, the three curves, key numbers) are generated. The two
reading paragraphs are written by hand from those numbers, so they are left as a stub.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.stress.scenarios import SCENARIO_LENGTHS

Row = dict[str, Any]

_TOKEN_BUCKETS = ((0, 999), (1000, 2999), (3000, 6999), (7000, 14999), (15000, None))
_CURVE_TURNS = (1, 3, 6, 10, 20)


def load_rows(path: Path) -> tuple[list[Row], int]:
    """Return ``(valid rows with numeric fields parsed, number of error rows)``."""
    rows: list[Row] = []
    errors = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if raw["error"]:
                errors += 1
                continue
            rows.append({
                **raw,
                "attachment_size_kb": int(raw["attachment_size_kb"]),
                "repeat": int(raw["repeat"]),
                "turn_index": int(raw["turn_index"]),
                "tokens_in": int(raw["tokens_in"]),
                "attachments_total_chars": int(raw["attachments_total_chars"]),
                "latency_ms": int(raw["latency_ms"]),
                "cost_usd": float(raw["cost_usd"]),
                "fact_recall": float(raw["fact_recall"]) if raw["fact_recall"] else None,
                "latency_ok": raw["latency_budget_passed"] == "True",
                "cost_ok": raw["cost_budget_passed"] == "True",
            })
    return rows, errors


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], body: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _sessions(rows: list[Row]) -> dict[tuple[str, int, int], list[Row]]:
    grouped: dict[tuple[str, int, int], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["attachment_size_kb"], row["repeat"])].append(row)
    return {key: sorted(group, key=lambda r: r["turn_index"]) for key, group in grouped.items()}


def summary_table(rows: list[Row]) -> str:
    cells: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        cells[(row["scenario"], row["attachment_size_kb"])].append(row)
    session_totals: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (scenario, size, _), turns in _sessions(rows).items():
        session_totals[(scenario, size)].append(sum(t["cost_usd"] for t in turns))

    body = []
    for key in sorted(cells):
        group = cells[key]
        latencies = [r["latency_ms"] for r in group]
        recalls = [r["fact_recall"] for r in group if r["fact_recall"] is not None]
        body.append([
            key[0], str(key[1]), str(len(group)),
            f"{percentile(latencies, 50):.0f}", f"{percentile(latencies, 95):.0f}",
            _fmt(_mean(session_totals[key]), 4),
            _fmt(100 * sum(r["cache_hit_kind"] == "exact" for r in group) / len(group), 1),
            _fmt(100 * sum(r["cache_hit_kind"] == "semantic" for r in group) / len(group), 1),
            _fmt(_mean(recalls), 3),
        ])
    return _table(
        ["Scenario", "KB", "n", "P50 ms", "P95 ms", "Cost/session $", "Exact hit %", "Semantic hit %", "Mean recall"],
        body,
    )


def latency_vs_tokens(rows: list[Row]) -> str:
    body = []
    for low, high in _TOKEN_BUCKETS:
        bucket = [r for r in rows if r["tokens_in"] >= low and (high is None or r["tokens_in"] <= high)]
        label = f"{low}+" if high is None else f"{low}-{high}"
        body.append([label, _fmt(statistics.median([r["latency_ms"] for r in bucket]) if bucket else None, 0), str(len(bucket))])
    return _table(["tokens_in bucket", "Median latency ms", "n"], body)


def _curve_columns(rows: list[Row]) -> list[tuple[str, int]]:
    sizes = sorted({r["attachment_size_kb"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows}, key=["growing", "pivot", "contradiction"].index)
    picked = [sizes[0]] + ([sizes[-1]] if len(sizes) > 1 else [])
    return [(scenario, size) for size in picked for scenario in scenarios]


def cumulative_cost_curve(rows: list[Row]) -> str:
    columns = _curve_columns(rows)
    sessions = _sessions(rows)
    cumulative: dict[tuple[str, int, int], float] = {}
    for key, turns in sessions.items():
        running = 0.0
        for turn in turns:
            running += turn["cost_usd"]
            cumulative[(*key, turn["turn_index"])] = running
    body = []
    for n in _CURVE_TURNS:
        cells = []
        for scenario, size in columns:
            values = [v for (s, z, _, t), v in cumulative.items() if (s, z, t) == (scenario, size, n)]
            cells.append(_fmt(_mean(values), 4))
        body.append([str(n)] + cells)
    return _table(["Turn"] + [f"{s} @{z}KB $" for s, z in columns], body)


def recall_vs_n(rows: list[Row]) -> str:
    columns = _curve_columns(rows)
    body = []
    for n in SCENARIO_LENGTHS:
        cells = []
        for scenario, size in columns:
            values = [
                r["fact_recall"] for r in rows
                if (r["scenario"], r["attachment_size_kb"], r["turn_index"]) == (scenario, size, n)
                and r["fact_recall"] is not None
            ]
            cells.append(_fmt(_mean(values), 3))
        body.append([str(n)] + cells)
    return _table(["N turns"] + [f"{s} @{z}KB" for s, z in columns], body)


def key_numbers(rows: list[Row]) -> list[str]:
    """Facts the written reading should cite."""
    out = []
    smallest = min(r["attachment_size_kb"] for r in rows)
    for scenario in sorted({r["scenario"] for r in rows}):
        base = [r for r in rows if r["scenario"] == scenario and r["attachment_size_kb"] == smallest]
        by_turn = {
            n: _mean([r["fact_recall"] for r in base if r["turn_index"] == n and r["fact_recall"] is not None])
            for n in sorted({r["turn_index"] for r in base})
        }
        broken = [n for n, value in by_turn.items() if value is not None and value < 1.0]
        last = max(by_turn) if by_turn else None
        out.append(
            f"{scenario} @{smallest}KB: recall first drops below 1.0 at turn "
            f"{broken[0] if broken else 'never'}; at turn {last} it is {_fmt(by_turn.get(last), 3)}."
        )
        drift = [r for r in base if r["memory_drift_passed"] != ""]
        if drift:
            alive = sum(r["memory_drift_passed"] == "True" for r in drift) / len(drift)
            out.append(f"{scenario} @{smallest}KB: turn-1 fact ({base[0]['tracked_fact']!r}) survives in {100 * alive:.1f}% of later turns.")

    out.append(
        f"Budget violations: latency {100 * sum(not r['latency_ok'] for r in rows) / len(rows):.1f}% of turns, "
        f"cost {100 * sum(not r['cost_ok'] for r in rows) / len(rows):.1f}% of turns."
    )
    for size in sorted({r["attachment_size_kb"] for r in rows}):
        cell = [r for r in rows if r["attachment_size_kb"] == size]
        out.append(
            f"{size}KB: median tokens_in {statistics.median(r['tokens_in'] for r in cell):.0f}, "
            f"mean attachment chars {statistics.fmean(r['attachments_total_chars'] for r in cell):.0f}, "
            f"latency budget met {100 * sum(r['latency_ok'] for r in cell) / len(cell):.1f}%, "
            f"cost budget met {100 * sum(r['cost_ok'] for r in cell) / len(cell):.1f}%."
        )
    return out


def render(rows: list[Row], errors: int) -> str:
    scenarios = sorted({r["scenario"] for r in rows})
    sizes = sorted({r["attachment_size_kb"] for r in rows})
    repeats = len({r["repeat"] for r in rows})
    key_lines = "\n".join(f"- {line}" for line in key_numbers(rows))
    return f"""# CAG Stress Test - Report

Generated by `uv run python -m evals.stress.report` from `results.csv`.

| Setting | Value |
|---|---|
| Scenarios | {", ".join(scenarios)} |
| Attachment sizes (KB) | {", ".join(map(str, sizes))} |
| Repeats per cell | {repeats} |
| Turns observed | {len(rows)} ({errors} error rows excluded) |

## 1. Summary table

One row per scenario x attachment size. `Cost/session` is the mean total over the repeats.
The 100 KB PDF holds ~102 K characters but each attachment is capped at
`MAX_ATTACHMENT_CHARS` = 60,000, so that column measures the truncated regime.
Conversational turns bypass both caches by design, so both hit rates are 0.

{summary_table(rows)}

## 2. Curves

### 2a. Latency vs `tokens_in`

{latency_vs_tokens(rows)}

### 2b. Cumulative cost vs turn (smallest and largest attachment)

{cumulative_cost_curve(rows)}

### 2c. Fact recall vs N

Share of facts declared in earlier turns still found in the summary, anchors or metadata
after turn N (`n/a` at N=1: nothing earlier to remember).

{recall_vs_n(rows)}

## 3. Reading: where the CAG starts to break

Key numbers to cite:

{key_lines}

**Paragraph 1 - where it breaks and which dimension dominates.** _Write from the numbers above._

**Paragraph 2 - what would justify moving to RAG.** _Write from the numbers above._
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "REPORT.md")
    args = parser.parse_args(argv)

    rows, errors = load_rows(args.csv)
    if not rows:
        print(f"No valid rows in {args.csv} ({errors} error rows).", file=sys.stderr)
        return 1
    args.output.write_text(render(rows, errors), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
