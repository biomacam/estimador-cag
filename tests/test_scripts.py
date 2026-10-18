"""Smoke tests for the runnable demo scripts under scripts/."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, f"scripts/{script}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_demo_cleaning_reports_the_expected_story() -> None:
    result = _run("demo_cleaning.py")

    assert result.returncode == 0, result.stderr
    assert "Files read: 6" in result.stdout
    assert "Rows after cleaning + dedup: 5" in result.stdout
    assert "valid_rows: 4" in result.stdout
    assert "discarded_rows: 1" in result.stdout
    assert "BUDGET-2024-0004" in result.stdout


def test_demo_pii_pseudonymizes_a_seed_transcript() -> None:
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("es_core_news_md"):
        pytest.skip("es_core_news_md is not installed")

    result = _run("demo_pii.py")

    assert result.returncode == 0, result.stderr
    assert "Mappings applied" in result.stdout
    assert "PERSON" in result.stdout


def test_preflight_runs_every_check() -> None:
    # /health and Postgres depend on a live stack, so only the local checks are asserted.
    result = _run("preflight.py")

    assert result.returncode in (0, 1)
    for name in ("Python version", "Required packages", "Corpus seed present", "Catalog validates"):
        assert f"[OK]   {name}" in result.stdout
    for name in ("Estimator /health", "Postgres + migration"):
        assert name in result.stdout
