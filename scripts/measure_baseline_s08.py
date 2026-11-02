#!/usr/bin/env python3
"""Measure SQL-side search latency over the benchmark queries.

Session 08 — Block 2.1 (baseline before the HNSW index) and Block 2.2 (re-run
right after creating it to expose the before/after gap).

Unlike ``query_examples.py`` (HTTP, timing includes the OpenAI embedding
round-trip), this embeds each query ONCE upfront and times only the SQL
execution: 1 warm-up run (discarded) plus 2 measured runs per query.

It runs the exact SELECT of the ``/search`` endpoint (``cosine_search_statement``).

Usage::

    docker compose run --rm estimator python scripts/measure_baseline_s08.py

At the end it prints the first query's embedding as a pgvector literal — paste
it into ``scripts/sql_s08/02_test_antipatron.sql`` for the psql demos.
"""

from __future__ import annotations

import asyncio
import statistics

from s08_common import (
    EmbeddedQuery,
    Stopwatch,
    embed_benchmark_queries,
    format_table,
    latency_summary,
    require_embedder,
    truncate,
    vector_literal,
)

from app.embedding_pipeline.search import cosine_search_statement
from app.persistence.database import get_async_session_factory

K = 5
WARMUP_RUNS = 1
MEASURED_RUNS = 2


async def measure(queries: list[EmbeddedQuery]) -> None:
    rows: list[list[str]] = []
    all_samples: list[float] = []

    async with get_async_session_factory()() as session:
        for _, query, vector in queries:
            samples: list[float] = []
            result_count = 0
            for run in range(WARMUP_RUNS + MEASURED_RUNS):
                with Stopwatch() as timer:
                    results = (await session.execute(cosine_search_statement(vector, K))).all()
                if run >= WARMUP_RUNS:
                    samples.append(timer.elapsed_ms)
                result_count = len(results)

            mean = statistics.fmean(samples)
            deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
            all_samples.extend(samples)
            rows.append([truncate(query, 40), f"{mean:.2f}", f"{deviation:.2f}", str(result_count)])

    print()
    print(format_table(["query", "mean_ms", "stdev_ms", "results"], rows))
    print()
    print(f"Summary ({len(all_samples)} measured runs): {latency_summary(all_samples)}")


def main() -> int:
    embedder = require_embedder()
    queries = embed_benchmark_queries(embedder)

    asyncio.run(measure(queries))

    _, query, vector = queries[0]
    print()
    print(f'pgvector literal for query 1 ("{truncate(query, 60)}"):')
    print(vector_literal(vector))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
