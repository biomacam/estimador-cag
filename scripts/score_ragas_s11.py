#!/usr/bin/env python3
"""Standalone RAGAS scorer for pre-collected Session 11 samples.

This script imports ONLY ragas + langchain-openai + datasets — never the app —
so it runs in an isolated venv, decoupled from the project's langchain stack
(``ragas`` imports ``langchain_community`` Vertex modules unconditionally at
load, which the project's current ``langchain-community`` no longer exposes).
Collect the samples first with::

    DATABASE_URL=... uv run python scripts/eval_ragas_s11.py --collect-only samples.json

then score them in the isolated venv::

    <ragas-venv>/python scripts/score_ragas_s11.py samples.json --out evals/ragas_baseline_s11.json

We never use Vertex (the judge is OpenAI), so lightweight stub modules are
registered before importing ragas to satisfy the import.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Judge: an OpenAI chat model; embeddings: ``text-embedding-3-small``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import types
from pathlib import Path

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _install_vertex_shims() -> None:
    """Satisfy ragas's hard Vertex imports without the Vertex SDK (unused here)."""
    with contextlib.suppress(Exception):
        import langchain_community.chat_models.vertexai  # noqa: F401

        return

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Vertex AI is not available in this scorer venv.")

    chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_mod.ChatVertexAI = _Unavailable
    sys.modules["langchain_community.chat_models.vertexai"] = chat_mod

    try:
        import langchain_community.llms as llms_mod

        if not hasattr(llms_mod, "VertexAI"):
            llms_mod.VertexAI = _Unavailable
    except Exception:  # noqa: BLE001
        llms_mod = types.ModuleType("langchain_community.llms")
        llms_mod.VertexAI = _Unavailable
        sys.modules["langchain_community.llms"] = llms_mod


def markdown_table(ids: list[str], rows: list[dict]) -> str:
    """One row per query plus an average row, with the four RAGAS metrics."""
    header = "| query | " + " | ".join(METRIC_NAMES) + " |"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 1)
    body = []
    for qid, row in zip(ids, rows, strict=True):
        cells = " | ".join(f"{row[m]:.3f}" for m in METRIC_NAMES)
        body.append(f"| {qid} | {cells} |")
    avg = {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
    avg_cells = " | ".join(f"{avg[m]:.3f}" for m in METRIC_NAMES)
    body.append(f"| **average** | {avg_cells} |")
    return "\n".join([header, sep, *body])


def run_ragas(
    samples: list[dict], judge_model: str, embedding_model: str
) -> list[dict]:
    """Compute the four RAGAS metrics over the samples; one score dict per sample."""
    _install_vertex_shims()

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    return [
        {
            m: (float(df.iloc[i][m]) if m in df.columns else float("nan"))
            for m in METRIC_NAMES
        }
        for i in range(len(samples))
    ]


def score_samples(
    samples: list[dict], judge_model: str, embedding_model: str, out: str | None
) -> str:
    """Score the samples, print the baseline table and optionally persist a report."""
    print(
        f"Scoring {len(samples)} samples (judge={judge_model}, embeddings={embedding_model})..."
    )
    rows = run_ragas(samples, judge_model, embedding_model)
    table = markdown_table([s["id"] for s in samples], rows)

    print("\n=== RAGAS generation baseline ===")
    print(table)

    if out:
        report = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "table_markdown": table,
            "per_query": [
                {"id": s["id"], **rows[i], "citation_report": s.get("citation_report")}
                for i, s in enumerate(samples)
            ],
        }
        Path(out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {out}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples", help="JSON written by eval_ragas_s11.py --collect-only"
    )
    parser.add_argument("--out", default=None, help="Write the scored report as JSON.")
    args = parser.parse_args()

    payload = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = payload["samples"]
    score_samples(
        samples,
        payload.get("judge_model", DEFAULT_JUDGE_MODEL),
        payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
