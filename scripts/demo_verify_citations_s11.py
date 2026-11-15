#!/usr/bin/env python3
"""Offline demo of line-level citation verification.

Builds a realistic grounded :class:`Estimate` from a small set of "retrieved"
chunks, deliberately plants ONE dangling citation (an id the LLM never saw) and
one line with no sufficient source data, then runs :func:`verify_citations` and
prints the :class:`CitationReport`. No network, no database — it exercises the
verification logic alone, so it doubles as the acceptance-criteria proof:

* every grounded line cites at least one real chunk,
* the verifier flags the planted dangling citation,
* the unsupported line is reported as "insufficient", not back-filled.

Usage::

    uv run python scripts/demo_verify_citations_s11.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.schemas import (
    Assumption,
    CitationReport,
    Estimate,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.rag.validation import verify_citations


def _chunk(chunk_id: int, content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=distance,
        budget_id="BUD-2024-001",
    )


def _retrieved() -> list[RetrievedChunk]:
    """The chunks the generator was actually given (ids 101, 102, 103)."""
    return [
        _chunk(
            101,
            "BUD-2024-001 :: AUTH-001 OAuth 2.0 authentication backend — 120 h",
            0.21,
        ),
        _chunk(
            102, "BUD-2024-001 :: PSD2-002 PSD2 open banking connectors — 160 h", 0.27
        ),
        _chunk(103, "BUD-2024-001 :: TXN-003 Transaction ledger service — 140 h", 0.31),
    ]


def _grounded(name: str, days: int, chunk_id: str, evidence: str) -> TaskItem:
    return TaskItem(
        name=name,
        engineer_days=days,
        grounded=True,
        sources=[
            SourceReference(
                chunk_id=chunk_id, document_id="BUD-2024-001", evidence=evidence
            )
        ],
    )


def _estimate() -> Estimate:
    """A grounded estimate with one planted dangling citation + one no-data line."""
    return Estimate(
        total_engineer_days=53,
        duration_weeks=11,
        modules=[
            WorkModule(
                name="Authentication & SCA",
                tasks=[
                    _grounded(
                        "OAuth 2.0 backend",
                        15,
                        "101",
                        "AUTH-001 OAuth 2.0 authentication backend — 120 h",
                    )
                ],
            ),
            WorkModule(
                name="PSD2 & Open Banking",
                tasks=[
                    _grounded(
                        "Open banking connectors",
                        20,
                        "102",
                        "PSD2-002 PSD2 open banking connectors — 160 h",
                    )
                ],
            ),
            WorkModule(
                name="Ledger",
                tasks=[
                    # Planted DANGLING citation: chunk 999 was never retrieved.
                    _grounded(
                        "Transaction ledger",
                        18,
                        "999",
                        "TXN-003 Transaction ledger service — 140 h",
                    )
                ],
            ),
            WorkModule(
                name="Reporting",
                tasks=[
                    # No sufficient source data: flagged, not back-filled with hours.
                    TaskItem(name="Regulatory reporting", grounded=False)
                ],
            ),
        ],
        assumptions=[
            Assumption(
                description="Regulatory reporting has no historical analog in the context.",
                impact="medium",
                rationale="No retrieved budget covers regulatory reporting.",
            )
        ],
        confidence="high",
        reasoning="Derived from the retrieved BUD-2024-001 components.",
    )


def _print_report(report: CitationReport) -> None:
    print("=== Citation verification report ===")
    print(
        f"lines: {report.total_lines}  grounded: {report.grounded_lines}  "
        f"dangling: {report.dangling_lines}  insufficient: {report.insufficient_lines}"
    )
    print(f"verified citations: {report.verified_citations}")
    print(f"dangling citations: {report.dangling_citations or '(none)'}")
    print()
    print(f"{'module':<24} {'line':<24} {'status':<13} cited -> dangling")
    print("-" * 88)
    for line in report.lines:
        print(
            f"{line.module:<24} {line.component:<24} {line.status:<13} "
            f"{line.cited_chunk_ids} -> {line.dangling_chunk_ids or '-'}"
        )


def main() -> int:
    retrieved_ids = {str(chunk.id) for chunk in _retrieved()}
    report = verify_citations(_estimate(), retrieved_ids)
    _print_report(report)

    ok = (
        report.dangling_citations == ["999"]
        and report.dangling_lines == 1
        and report.grounded_lines == 2
        and report.insufficient_lines == 1
    )
    print()
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
