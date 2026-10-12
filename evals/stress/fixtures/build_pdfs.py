"""Generate calibrated synthetic PDFs for the stress runner.

Produces ``attach_{5,20,50,100}kb.pdf`` under this directory. The suffix refers to
the *extracted text size* (what ``pypdf`` pulls out), not the bytes on disk: the
runner cares about the prompt budget. ``attach_100kb`` exceeds
``MAX_ATTACHMENT_CHARS`` (60,000) on purpose, to observe how truncation behaves.

Deterministic: same paragraph, same repeats, fixed creation date, so re-running
yields byte-identical PDFs. The PDFs are gitignored; only this script is committed.

Run with::

    uv run python -m evals.stress.fixtures.build_pdfs
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

TARGETS_KB = (5, 20, 50, 100)

_PARAGRAPH = (
    "The supplier portal must replace the existing manual workflow. Vendors "
    "upload PDFs of compliance certificates which the procurement team reviews "
    "for completeness, expiration, and conformance to the company's standard. "
    "Approved suppliers are unlocked for purchase order generation; rejected "
    "suppliers must remediate before the next review cycle. Every state "
    "transition is recorded in an immutable audit log retained for seven years "
    "in compliance with internal policy. Reviewers may attach comments to a "
    "specific certificate; comments are visible to the supplier and form part "
    "of the formal record. The portal must integrate with the legacy ERP for "
    "vendor master data synchronisation, propagating changes within fifteen "
    "minutes. Single sign-on is mandatory for internal users; vendors use "
    "magic-link email authentication scoped to a single tenant. Performance "
    "expectations are P95 page load under two seconds with two hundred "
    "concurrent users, and the system must remain available under one hour of "
    "downtime per quarter excluding planned maintenance windows scheduled "
    "outside business hours. "
)

_FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def pdf_path(kb: int, directory: Path | None = None) -> Path:
    return (directory or Path(__file__).parent) / f"attach_{kb}kb.pdf"


def _build_pdf(target_kb: int, output_path: Path) -> tuple[int, int]:
    """Write a PDF with ~``target_kb`` KB of text; return ``(text_chars, file_bytes)``."""
    pdf = FPDF()
    pdf.set_creation_date(_FIXED_DATE)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    target_chars = target_kb * 1024
    text_chars = 0
    while text_chars < target_chars:
        pdf.multi_cell(0, 5, _PARAGRAPH)
        text_chars += len(_PARAGRAPH)
        if text_chars < target_chars:
            pdf.ln(3)

    pdf.output(str(output_path))
    return text_chars, output_path.stat().st_size


def _verify_extracted(path: Path) -> int:
    """Characters ``pypdf`` extracts, i.e. what the attachment pipeline will see."""
    reader = PdfReader(str(path))
    return len("\n\n".join(page.extract_text() or "" for page in reader.pages))


def build_all(directory: Path | None = None) -> list[Path]:
    """(Re)generate every fixture and return their paths."""
    directory = directory or Path(__file__).parent
    paths = []
    for kb in TARGETS_KB:
        path = pdf_path(kb, directory)
        text_chars, file_bytes = _build_pdf(kb, path)
        print(
            f"  {path.name}: text_chars={text_chars} file_bytes={file_bytes} "
            f"pypdf_extracted={_verify_extracted(path)}"
        )
        paths.append(path)
    return paths


def main() -> int:
    print(f"Writing PDFs to: {Path(__file__).parent}")
    build_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
