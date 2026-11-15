"""Session 11 RAGAS baseline: the pure pieces (answer rendering, metrics table)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ragas_s11 import render_estimate_text
from score_ragas_s11 import METRIC_NAMES, markdown_table

from app.rag.schemas import (
    Assumption,
    Estimate,
    SourceReference,
    TaskItem,
    WorkModule,
)


def test_render_estimate_text_tags_grounded_and_no_data_lines():
    estimate = Estimate(
        total_engineer_days=15,
        duration_weeks=3,
        modules=[
            WorkModule(
                name="Auth",
                tasks=[
                    TaskItem(
                        name="OAuth backend",
                        engineer_days=15,
                        grounded=True,
                        sources=[
                            SourceReference(
                                chunk_id="1",
                                document_id="BUD-2024-001",
                                evidence="120 hours",
                            )
                        ],
                    ),
                    TaskItem(name="Regulatory reporting", grounded=False),
                ],
            )
        ],
        assumptions=[
            Assumption(description="Reporting scope", impact="low", rationale="r")
        ],
        confidence="medium",
        reasoning="From BUD-2024-001.",
    )

    text = render_estimate_text(estimate)

    assert "Total: 15 engineer-days." in text
    assert "- OAuth backend: 15d [grounded]" in text
    assert "- Regulatory reporting: n/a [no-data]" in text
    assert "Assumptions: Reporting scope" in text


def test_render_estimate_text_for_insufficient_context_has_no_numbers():
    estimate = Estimate(
        confidence="insufficient",
        reasoning="r",
        insufficient_context_explanation="No relevant budgets.",
    )

    assert render_estimate_text(estimate) == (
        "Insufficient context to produce an estimate. No relevant budgets."
    )


def test_markdown_table_has_one_row_per_query_and_an_average_row():
    rows = [
        dict.fromkeys(METRIC_NAMES, 0.2),
        {**dict.fromkeys(METRIC_NAMES, 0.6), "context_recall": 1.0},
    ]

    table = markdown_table(["Q1", "Q2"], rows).splitlines()

    assert table[0] == "| query | " + " | ".join(METRIC_NAMES) + " |"
    assert [line.split("|")[1].strip() for line in table[2:]] == [
        "Q1",
        "Q2",
        "**average**",
    ]
    assert table[-1] == "| **average** | 0.400 | 0.400 | 0.400 | 0.600 |"
