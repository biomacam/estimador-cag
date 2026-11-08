#!/usr/bin/env python3
"""Measure retrieval quality across the four search/rerank configurations.

Runs the golden set (``evals/golden_retrieval.json``) through the same
``retrieve()`` pipeline the API uses, for each of:

    A  Vector  / no rerank   (baseline)
    B  Hybrid  / no rerank
    C  Vector  / rerank
    D  Hybrid  / rerank

and saves precision@5 (mean over the queries) and query latency (mean over
measured runs) per configuration, plus raw rankings and a Markdown comparison.
The database must contain exactly the base corpus's budget-component chunks.

Method notes:
* Queries are embedded ONCE upfront; embedding latency is an OpenAI round-trip
  shared by every config, so it is excluded from the timings (we measure the
  retrieval/rerank cost the techniques actually add).
* A permissive distance threshold is used so the top-5 is never truncated by the
  relevance floor: we are comparing RANKING quality, not the soft-fail gate.
* A retrieved chunk is relevant iff its parent ``budget_id`` is in the query's
  annotated ``relevant_budget_ids``. precision@5 = relevant_in_top5 / 5.

Usage (host, Postgres up + migrations applied + base corpus ingested +
OPENAI_API_KEY in .env)::

    uv run python scripts/eval_retrieval_s10.py

Hybrid needs migration 0006 (``alembic upgrade head``) for English full-text search.
Reranking runs download the
cross-encoder weights on first use; verify first with
``python -m app.rag.retrieval.verify_reranker``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from s08_common import Stopwatch, require_embedder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dependencies import get_reranker  # noqa: E402
from app.rag.retrieval.pipeline import retrieve  # noqa: E402
from app.embedding_pipeline.chunker import JSONStructuralChunker  # noqa: E402
from app.embedding_pipeline.embedder import MODEL  # noqa: E402
from app.embedding_pipeline.schemas import Budget  # noqa: E402
from app.persistence.database import get_async_session_factory  # noqa: E402
from app.persistence.models import ChunkRow  # noqa: E402
from sqlalchemy import select, text  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# (id, search label, rerank label, search_mode, rerank)
CONFIGS = [
    ("A", "Vector", "No", "vector", False),
    ("B", "Hybrid", "No", "hybrid", False),
    ("C", "Vector", "Yes", "vector", True),
    ("D", "Hybrid", "Yes", "hybrid", True),
]

# No effective relevance floor: a full top-k is needed to grade ranking quality.
NO_FLOOR_THRESHOLD = 2.0
# Latency sampling per (query, config): one discarded warm-up + N measured runs.
MEASURED_RUNS = 3


def precision_at_k(chunks, relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results whose budget is genuinely relevant."""
    if k <= 0:
        raise ValueError("Precision cutoff k must be positive.")
    hits = sum(1 for chunk in chunks[:k] if chunk.budget_id in relevant_ids)
    return hits / k


async def _run_once(query_embedding, query_text, search_mode, rerank, settings, chunk_types, k):
    return await retrieve(
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=k,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=k,
        distance_threshold=NO_FLOOR_THRESHOLD,
        rrf_k=settings.RRF_K,
        chunk_types=chunk_types,
    )


async def verify_corpus(corpus_path: Path, chunk_types: list[str]) -> int:
    """Reject incomplete or contaminated corpora before publishing measurements."""
    budgets = [
        Budget.model_validate(budget)
        for budget in json.loads(corpus_path.read_text(encoding="utf-8"))
    ]
    expected = sorted(
        (chunk.metadata["budget_id"], chunk.text)
        for chunk in JSONStructuralChunker().chunk(budgets)
    )
    async with get_async_session_factory()() as session:
        rows = (await session.execute(
            select(ChunkRow.content, ChunkRow.metadata_json, ChunkRow.embedding.is_not(None))
            .where(ChunkRow.chunk_type.in_(chunk_types))
        )).all()
    actual = sorted((row[1].get("budget_id", ""), row[0]) for row in rows)
    if actual != expected or not all(row[2] for row in rows):
        raise ValueError(
            "Evaluation requires the complete base corpus with embeddings and no additional "
            "budget-component chunks. Use a dedicated database and ingest data/budgets_sample.json."
        )
    return len(rows)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evals" / "retrieval_results.json",
        help="Save raw measurements as JSON and a companion Markdown comparison.",
    )
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must have a .json suffix.")
    settings = get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    chunk_types = golden["chunk_types"]
    k = golden["k"]
    if len(queries) != 5 or k != 5:
        raise ValueError("This benchmark requires exactly five queries and k=5.")
    corpus_path = ROOT / golden["corpus"]
    corpus_chunks = await verify_corpus(corpus_path, chunk_types)
    async with get_async_session_factory()() as session:
        indexes = (await session.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks' ORDER BY indexname")
        )).all()
        lexical_hits = {
            query["id"]: await session.scalar(
                select(text("count(*)")).select_from(ChunkRow)
                .where(ChunkRow.chunk_type.in_(chunk_types))
                .where(text("content_tsv @@ plainto_tsquery('english', :query)"))
                .params(query=query["query"])
            )
            for query in queries
        }

    embedder = require_embedder()
    print(f"Embedding {len(queries)} golden queries (excluded from timings)...")
    embeddings = {q["id"]: embedder.embed_one(q["query"]) for q in queries}

    # Warm the reranker once so its one-time model load does not skew the first timed run.
    print("Warming up the cross-encoder (first load downloads weights)...")
    get_reranker().load()

    results = {cfg[0]: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS}

    for cfg_id, _s, _r, search_mode, rerank in CONFIGS:
        for q in queries:
            relevant = set(q["relevant_budget_ids"])
            emb = embeddings[q["id"]]

            # Warm-up (discarded) then measured runs.
            await _run_once(emb, q["query"], search_mode, rerank, settings, chunk_types, k)
            samples = []
            runs = []
            last = None
            for _ in range(MEASURED_RUNS):
                with Stopwatch() as sw:
                    last = await _run_once(
                        emb, q["query"], search_mode, rerank, settings, chunk_types, k
                    )
                samples.append(sw.elapsed_ms)
                runs.append({
                    "latency_ms": sw.elapsed_ms,
                    "precision_at_5": precision_at_k(last.chunks, relevant, k),
                    "chunk_ids": [chunk.id for chunk in last.chunks],
                    "budget_ids": [chunk.budget_id for chunk in last.chunks],
                })

            if not last.chunks:
                raise ValueError(f"No results for configuration {cfg_id}, query {q['id']}.")
            precision = statistics.fmean(run["precision_at_5"] for run in runs)
            results[cfg_id]["precisions"].append(precision)
            results[cfg_id]["latencies_ms"].extend(samples)
            results[cfg_id]["per_query"][q["id"]] = {
                "precision_at_5": precision,
                "latency_ms": statistics.fmean(samples),
                "runs": runs,
            }

    comparison = format_report(results, queries, k, lexical_hits)
    print(comparison)
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "golden_sha256": hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "corpus_chunks": corpus_chunks,
        "lexical_hits_by_query": lexical_hits,
        "queries": queries,
        "method": {
            "k": k, "measured_runs_per_query": MEASURED_RUNS, "warmup_runs_per_query": 1,
            "latency_scope": "Retrieval plus reranking; excludes embedding, model loading and HTTP.",
            "recall_k": settings.RETRIEVAL_RECALL_TOP_K, "rrf_k": settings.RRF_K,
            "distance_threshold": NO_FLOOR_THRESHOLD,
            "embedding_model": MODEL, "reranker_model": settings.RERANKER_MODEL,
            "text_search_config": "english", "python": platform.python_version(),
            "platform": platform.platform(),
            "indexes": [dict(name=row[0], definition=row[1]) for row in indexes],
        },
        "configurations": {
            cfg_id: {
                "search_mode": mode, "rerank": rerank,
                "precision_at_5": statistics.fmean(results[cfg_id]["precisions"]),
                "latency_ms": statistics.fmean(results[cfg_id]["latencies_ms"]),
                "per_query": results[cfg_id]["per_query"],
            }
            for cfg_id, _s, _r, mode, rerank in CONFIGS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(comparison, encoding="utf-8")
    print(f"Saved measurements to {args.output}")
    return 0


def format_report(
    results: dict, queries: list, k: int, lexical_hits: dict[str, int] | None = None,
) -> str:
    lines = [
        f"# Retrieval evaluation: precision@{k} and latency\n",
        "Latency includes retrieval and reranking, but excludes embeddings, model loading "
        "and HTTP. Each query/configuration has one discarded warm-up and three measured runs.\n",
        "Relevance is budget-level, precision is chunk-level: multiple chunks from one relevant "
        "budget count individually. Q2/Q3 have only four relevant chunks and Q4 only three, "
        "so their maximum precision@5 is 0.8/0.8/0.6.\n",
        "The shipped corpus, queries and full-text configuration are English.\n",
        f"| Config | Search | Reranking | Precision@{k} | Latency (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cfg_id, search_label, rerank_label, _m, _rr in CONFIGS:
        bucket = results[cfg_id]
        mean_p = statistics.fmean(bucket["precisions"])
        mean_l = statistics.fmean(bucket["latencies_ms"])
        lines.append(f"| {cfg_id} | {search_label} | {rerank_label} | {mean_p:.2f} | {mean_l:.1f} |")

    lines.extend([
        f"\n## Per-query precision@{k}\n",
        "| Query | " + " | ".join(cfg[0] for cfg in CONFIGS) + " |",
        "| --- | " + " | ".join("---" for _ in CONFIGS) + " |",
    ])
    for q in queries:
        row = [q["id"]] + [
            f"{results[cfg[0]]['per_query'][q['id']]['precision_at_5']:.2f}" for cfg in CONFIGS
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["\n## Golden set annotations\n", "| Query | Project | Relevant budgets |", "| --- | --- | --- |"])
    for query in queries:
        lines.append(
            f"| {query['id']} | {query['query']} | {', '.join(query['relevant_budget_ids'])} |"
        )
    if lexical_hits is not None:
        lines.extend([
            "\n## Lexical branch diagnostics\n",
            "| Query | Matching chunks |", "| --- | --- |",
        ])
        for query in queries:
            lines.append(f"| {query['id']} | {lexical_hits[query['id']]} |")
        if not any(lexical_hits.values()):
            lines.append(
                "\nNo lexical matches were found. Hybrid rankings therefore equal vector "
                "rankings before reranking. plainto_tsquery requires every non-stop-word "
                "term in each long project description to match in a single chunk."
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
