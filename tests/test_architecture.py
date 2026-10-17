"""Tests for the CAG / RAG / hybrid architecture decision."""

from __future__ import annotations

import subprocess
import sys

from app.ingestion.architecture import (
    CURRENT_MODEL,
    PROYECTO_2,
    CAGViability,
    CorpusProfile,
    ModelProfile,
    assess_cag_viability,
    recommend_architecture,
)


def _corpus(**overrides) -> CorpusProfile:
    values = dict(
        name="corpus",
        estimated_tokens=20_000,
        refresh_frequency_days=30,
        traceability_required=False,
        access_control_required=False,
    )
    return CorpusProfile(**{**values, **overrides})


def test_proyecto_2_recommends_rag() -> None:
    # 250K tokens do not fit the usable window, and it needs traceability and access control.
    assert recommend_architecture(PROYECTO_2, CURRENT_MODEL) == "RAG"


def test_small_stable_unrestricted_corpus_recommends_cag() -> None:
    assert recommend_architecture(_corpus(), CURRENT_MODEL) == "CAG"


def test_fits_and_is_stable_but_needs_traceability_recommends_hybrid() -> None:
    assert recommend_architecture(_corpus(traceability_required=True), CURRENT_MODEL) == "Hybrid"


def test_fits_and_is_stable_but_needs_access_control_recommends_hybrid() -> None:
    assert recommend_architecture(_corpus(access_control_required=True), CURRENT_MODEL) == "Hybrid"


def test_volatile_corpus_breaks_prefix_caching() -> None:
    assert recommend_architecture(_corpus(refresh_frequency_days=1), CURRENT_MODEL) == "RAG"


def test_window_boundary_is_70_percent() -> None:
    model = ModelProfile("m", context_window=100_000, cost_per_1k_input=0.0, prefix_caching=True)

    assert recommend_architecture(_corpus(estimated_tokens=70_000), model) == "CAG"
    assert recommend_architecture(_corpus(estimated_tokens=70_001), model) == "RAG"


def test_cag_viability_is_an_and() -> None:
    assert CAGViability(True, True, True, True).viable is True
    assert CAGViability(True, True, True, True).failing_constraints() == []

    one_red = CAGViability(
        context_window_ok=False, cost_ok=True, latency_ok=True, lost_in_the_middle_ok=True
    )
    assert one_red.viable is False
    assert one_red.failing_constraints() == ["context_window"]


def test_assess_proyecto_2_fails_the_volume_axes() -> None:
    viability = assess_cag_viability(PROYECTO_2, CURRENT_MODEL)

    assert viability.viable is False
    assert {"context_window", "latency", "lost_in_the_middle"} <= set(
        viability.failing_constraints()
    )


def test_a_model_without_prefix_caching_fails_the_cost_constraint() -> None:
    model = ModelProfile("m", context_window=128_000, cost_per_1k_input=0.0, prefix_caching=False)

    assert "cost" in assess_cag_viability(_corpus(), model).failing_constraints()


def test_cli_prints_the_recommendation() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "app.ingestion.architecture"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "Architecture recommendation: RAG" in out
    assert "NOT viable" in out
