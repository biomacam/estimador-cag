#!/usr/bin/env python3
"""Sweep ``hnsw.ef_search`` and measure its impact on latency and recall.

Session 08 — Block 3.1. Draws the two curves (recall saturates, latency keeps
growing) used to pick a reasoned ``ef_search`` value for the corpus.

1. Ground truth: each benchmark query runs with ``SET LOCAL enable_indexscan
   = off`` (forced sequential scan, exact → recall = 1.0) to capture the true
   top-10 chunk ids.
2. For each ``ef_search`` in [10, 20, 40, 80, 120, 200]: ``SET LOCAL
   hnsw.ef_search = N`` inside one transaction, run the queries (1 warm-up +
   2 measured runs each), capture latency and top-5 ids.
3. Recall per query = |index top-5 ∩ ground-truth top-5| / 5.
4. ★ marks the smallest value whose recall is within 0.005 of the best
   observed (or ≥ 0.99).

``SET LOCAL`` only affects the issuing transaction, so the script owns the SQL
session instead of calling ``/search``.

Requires the HNSW index (``scripts/sql_s08/01_create_hnsw.sql``).

Usage::

    docker compose run --rm estimator python scripts/sweep_ef_search_s08.py
"""

from __future__ import annotations

import asyncio
import statistics

from sqlalchemy import text

from s08_common import (
    EmbeddedQuery,
    Stopwatch,
    embed_benchmark_queries,
    format_table,
    require_embedder,
)

from app.embedding_pipeline.search import cosine_search_statement
from app.persistence.database import get_async_session_factory

EF_SEARCH_VALUES = [10, 20, 40, 80, 120, 200]
PGVECTOR_DEFAULT_EF = 40
K = 5
GROUND_TRUTH_K = 10
WARMUP_RUNS = 1
MEASURED_RUNS = 2
RECALL_TOLERANCE = 0.005


async def ground_truth(
    session_factory, queries: list[EmbeddedQuery]
) -> tuple[dict[str, list[int]], float]:
    """Exact top-10 ids per query via forced sequential scan (recall = 1.0)."""
    truth: dict[str, list[int]] = {}
    latencies: list[float] = []
    async with session_factory() as session, session.begin():
        await session.execute(text("SET LOCAL enable_indexscan = off"))
        for _, query, vector in queries:
            with Stopwatch() as timer:
                rows = (await session.execute(cosine_search_statement(vector, GROUND_TRUTH_K))).all()
            latencies.append(timer.elapsed_ms)
            truth[query] = [row.chunk_id for row in rows]
    return truth, statistics.fmean(latencies)


async def sweep_one(
    session_factory,
    queries: list[EmbeddedQuery],
    truth: dict[str, list[int]],
    ef_search: int,
) -> tuple[float, float]:
    """Return (avg_latency_ms, avg_recall) for one ef_search value."""
    latencies: list[float] = []
    recalls: list[float] = []
    async with session_factory() as session, session.begin():
        # SET takes no bind parameters; ef_search comes from a trusted int constant.
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        for _, query, vector in queries:
            samples: list[float] = []
            top_ids: list[int] = []
            for run in range(WARMUP_RUNS + MEASURED_RUNS):
                with Stopwatch() as timer:
                    rows = (await session.execute(cosine_search_statement(vector, K))).all()
                if run >= WARMUP_RUNS:
                    samples.append(timer.elapsed_ms)
                top_ids = [row.chunk_id for row in rows]
            latencies.append(statistics.fmean(samples))
            expected = set(truth[query][:K])
            recalls.append(len(expected & set(top_ids)) / K)
    return statistics.fmean(latencies), statistics.fmean(recalls)


def recommendation_labels(results: list[tuple[int, float, float]]) -> dict[int, str]:
    """Compute the recommendation column: ★ at the recall/latency sweet spot."""
    best_recall = max(recall for _, _, recall in results)
    threshold = min(0.99, best_recall - RECALL_TOLERANCE)
    recommended = next(
        (ef for ef, _, recall in results if recall >= threshold),
        results[-1][0],
    )

    labels: dict[int, str] = {}
    for ef, _, recall in results:
        if ef == recommended:
            label = "★ recommended"
        elif ef < recommended:
            label = "fast / low quality" if recall < 0.95 else "good"
        else:
            label = "diminishing returns"
        if ef == PGVECTOR_DEFAULT_EF:
            label += " (pgvector default)"
        labels[ef] = label
    return labels


async def run_sweep() -> None:
    embedder = require_embedder()
    queries = embed_benchmark_queries(embedder)
    session_factory = get_async_session_factory()

    print("Step 1/2: ground truth via forced sequential scan (recall = 1.0)...")
    truth, seq_latency = await ground_truth(session_factory, queries)
    print(f"  sequential scan avg latency: {seq_latency:.1f} ms (the no-index reference)")

    print(f"Step 2/2: sweeping hnsw.ef_search over {EF_SEARCH_VALUES}...")
    results: list[tuple[int, float, float]] = []
    for ef_search in EF_SEARCH_VALUES:
        latency, recall = await sweep_one(session_factory, queries, truth, ef_search)
        results.append((ef_search, latency, recall))
        print(f"  ef_search={ef_search:<4} avg_latency={latency:7.2f} ms  avg_recall={recall:.3f}")

    labels = recommendation_labels(results)
    rows = [
        [str(ef), f"{latency:.2f}", f"{recall:.3f}", labels[ef]] for ef, latency, recall in results
    ]
    print()
    print(format_table(["ef_search", "avg_latency_ms", "avg_recall", "recommendation"], rows))
    print()
    print(
        "Reading the curves: recall saturates while latency keeps growing — "
        "pick the smallest ef_search past the recall knee."
    )


def main() -> int:
    asyncio.run(run_sweep())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
