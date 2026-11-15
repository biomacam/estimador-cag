#!/usr/bin/env python3
"""RAGAS generation-quality baseline for the Session 11 golden set.

For each query in ``evals/golden_generation_s11.json`` this runs the REAL RAG
generation pipeline (reformulate -> embed -> retrieve -> assemble -> generate)
and records the four inputs RAGAS needs:

    question      -> the estimation request (golden ``query``)
    answer        -> the estimate the pipeline generated, rendered as text
    contexts      -> the retrieved chunks placed in the context block
    ground_truth  -> the expert reference estimate (golden ``ground_truth``)

It then computes the four RAGAS metrics — faithfulness, answer_relevancy,
context_precision, context_recall — with an OpenAI chat model as judge and
``text-embedding-3-small`` for the embedding-based metrics, and prints a table
with one row per query plus an average row: the generation-quality BASELINE.

Prerequisites (host run):
    * .env with OPENAI_API_KEY
    * the stack up (docker compose up -d) and the budget corpus ingested
    * ``DATABASE_URL`` pointing at the stack's Postgres from the host, e.g.
      postgresql+psycopg://estimator:estimator@localhost:5432/estimator
    * ragas + datasets + langchain-openai available in the venv that scores
      (see ``score_ragas_s11.py`` for the isolated-venv flow)

Usage::

    uv run python scripts/eval_ragas_s11.py --collect-only samples.json
    <ragas-venv>/python scripts/score_ragas_s11.py samples.json --out evals/ragas_baseline_s11.json

    # one-venv run (ragas installed alongside the app):
    uv run python scripts/eval_ragas_s11.py --out evals/ragas_baseline_s11.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_ragas_s11 import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    score_samples,
)

from app.config import get_settings
from app.dependencies import get_embedder, get_token_encoder
from app.rag.context_assembler import build_context_block, truncate_to_token_budget
from app.rag.estimator import generate_estimate
from app.rag.query_reformulator import compose_search_text, reformulate_query
from app.rag.retrieval.pipeline import retrieve
from app.rag.schemas import Estimate
from app.rag.validation import verify_citations

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"
# Permissive threshold: the generator must see context for every query so RAGAS
# grades GENERATION, not the soft-fail gate.
GENERATION_THRESHOLD = 1.2


def render_estimate_text(estimate: Estimate) -> str:
    """Render an :class:`Estimate` as the plain-text 'answer' RAGAS scores."""
    if estimate.confidence == "insufficient":
        return (
            "Insufficient context to produce an estimate. "
            f"{estimate.insufficient_context_explanation or ''}"
        ).strip()

    lines: list[str] = []
    if estimate.total_engineer_days is not None:
        lines.append(f"Total: {estimate.total_engineer_days} engineer-days.")
    if estimate.duration_weeks is not None:
        lines.append(f"Duration: {estimate.duration_weeks} weeks.")
    lines.append(f"Confidence: {estimate.confidence}.")
    for module in estimate.modules:
        lines.append(f"Module {module.name}:")
        for task in module.tasks:
            days = "n/a" if task.engineer_days is None else f"{task.engineer_days}d"
            tag = "grounded" if task.grounded else "no-data"
            lines.append(f"  - {task.name}: {days} [{tag}]")
    if estimate.assumptions:
        lines.append(
            "Assumptions: " + "; ".join(a.description for a in estimate.assumptions)
        )
    lines.append(f"Reasoning: {estimate.reasoning}")
    return "\n".join(lines)


async def collect_sample(item: dict, embedder) -> dict:
    """Run the pipeline for one golden item and return a RAGAS-ready sample."""
    settings = get_settings()
    transcript = item.get("transcript") or item["query"]

    query = await reformulate_query(transcript)
    search_text = compose_search_text(query)
    embedding = await asyncio.to_thread(embedder.embed_one, search_text)

    retrieval = await retrieve(
        query_embedding=embedding,
        query_text=search_text,
        search_mode="hybrid",
        rerank=True,
        top_k=settings.RETRIEVAL_TOP_K,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=settings.RERANK_TOP_N,
        distance_threshold=GENERATION_THRESHOLD,
        rrf_k=settings.RRF_K,
    )

    kept = truncate_to_token_budget(
        retrieval.chunks, settings.MAX_CONTEXT_TOKENS, get_token_encoder()
    )
    context_block = build_context_block(kept)
    estimate = await generate_estimate(context_block, structured_query=query)
    report = verify_citations(estimate, {str(chunk.id) for chunk in kept})

    return {
        "id": item["id"],
        "question": item["query"],
        "answer": render_estimate_text(estimate),
        "contexts": [chunk.content for chunk in kept],
        "ground_truth": item["ground_truth"],
        "citation_report": report.model_dump(),
    }


async def _amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N queries."
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Write the scored report as JSON."
    )
    parser.add_argument(
        "--collect-only",
        type=str,
        default=None,
        metavar="PATH",
        help="Only run the app pipeline and dump the samples to PATH (no RAGAS); score "
        "them afterwards with score_ragas_s11.py in an isolated venv.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help="Golden generation set (default: evals/golden_generation_s11.json).",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    judge_model = golden.get("judge_model", DEFAULT_JUDGE_MODEL)
    embedding_model = golden.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    items = golden["queries"][: args.limit] if args.limit else golden["queries"]

    embedder = get_embedder()
    if embedder is None:
        print(
            "ERROR: no OpenAI client available (OPENAI_API_KEY missing in .env).",
            file=sys.stderr,
        )
        return 1

    print(f"Collecting {len(items)} samples through the generation pipeline...")
    samples = []
    for item in items:
        print(f"  - {item['id']}: {item['query'][:70]}...")
        samples.append(await collect_sample(item, embedder))

    if args.collect_only:
        Path(args.collect_only).write_text(
            json.dumps(
                {
                    "judge_model": judge_model,
                    "embedding_model": embedding_model,
                    "samples": samples,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"\nWrote {len(samples)} samples to {args.collect_only} (score with score_ragas_s11.py)."
        )
        return 0

    score_samples(samples, judge_model, embedding_model, args.out)
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
