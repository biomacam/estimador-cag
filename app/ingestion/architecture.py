"""Architecture decision: CAG, RAG or hybrid?

Executable form of the decision. It is *not* wired into the FastAPI request path:
like ``catalog/inspect.py`` it is a CLI you run to reason about a corpus before any
ingestion happens::

    python -m app.ingestion.architecture

Two pieces:

* :class:`CAGViability`: the four CAG *constraints* (context window, cost, latency,
  lost-in-the-middle). ``viable`` is an ``all([...])`` on purpose: the four are an
  **AND**, not an OR, so one red constraint kills CAG.
* :func:`recommend_architecture`: evaluates the four *decision axes* (volume,
  refresh frequency, traceability, access control) and returns
  ``"CAG" | "Hybrid" | "RAG"``.

Fine-tuning is deliberately absent: it is an orthogonal layer (it teaches *how* to
answer, it does not add *new data* nor solve traceability).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

# Below this many tokens we assume the model still "uses the middle" and stays
# under a reasonable latency SLO. Above it, lost-in-the-middle and latency bite.
_CENTER_RECALL_TOKEN_BUDGET = 100_000
# A corpus fits CAG only if it leaves headroom in the window (we use 70%).
_USABLE_WINDOW_FRACTION = 0.7
# Prefix caching only pays off if the corpus is stable for at least a week.
_CACHE_FRIENDLY_REFRESH_DAYS = 7


@dataclass
class CAGViability:
    """The four CAG constraints. ``viable`` is an AND of all four."""

    context_window_ok: bool  # does the whole corpus fit in the context window?
    cost_ok: bool  # is the per-turn cost with prefix caching affordable?
    latency_ok: bool  # does latency with the whole corpus stay under the SLO?
    lost_in_the_middle_ok: bool  # do benchmarks show the model USES the middle?

    @property
    def viable(self) -> bool:
        return all(
            [
                self.context_window_ok,
                self.cost_ok,
                self.latency_ok,
                self.lost_in_the_middle_ok,
            ]
        )

    def failing_constraints(self) -> list[str]:
        """Names of the constraints in red; empty iff ``viable``."""
        return [
            name
            for name, ok in (
                ("context_window", self.context_window_ok),
                ("cost", self.cost_ok),
                ("latency", self.latency_ok),
                ("lost_in_the_middle", self.lost_in_the_middle_ok),
            )
            if not ok
        ]


@dataclass
class CorpusProfile:
    name: str
    estimated_tokens: int  # total volume
    refresh_frequency_days: float  # how often the data changes
    traceability_required: bool  # must we be able to cite the source?
    access_control_required: bool  # must we filter by per-user permissions?


@dataclass
class ModelProfile:
    name: str
    context_window: int  # tokens
    cost_per_1k_input: float  # USD
    prefix_caching: bool  # does it support prefix caching?


def assess_cag_viability(corpus: CorpusProfile, model: ModelProfile) -> CAGViability:
    """Derive the four CAG constraints from the corpus and model profiles.

    Deliberately simple heuristics: the point is clarity, not a calibrated cost model.
    """
    fits = corpus.estimated_tokens <= model.context_window * _USABLE_WINDOW_FRACTION
    return CAGViability(
        context_window_ok=fits,
        cost_ok=model.prefix_caching
        and corpus.refresh_frequency_days >= _CACHE_FRIENDLY_REFRESH_DAYS,
        latency_ok=corpus.estimated_tokens <= _CENTER_RECALL_TOKEN_BUDGET,
        lost_in_the_middle_ok=corpus.estimated_tokens <= _CENTER_RECALL_TOKEN_BUDGET,
    )


def recommend_architecture(
    corpus: CorpusProfile, model: ModelProfile
) -> Literal["CAG", "Hybrid", "RAG"]:
    """Recommend an architecture over the four decision axes."""
    # Axis 1, volume: does it fit with headroom in the window?
    fits_in_window = corpus.estimated_tokens <= model.context_window * _USABLE_WINDOW_FRACTION
    # Axis 2, refresh: changes more often than weekly break prefix caching.
    cache_friendly = corpus.refresh_frequency_days >= _CACHE_FRIENDLY_REFRESH_DAYS
    # Axis 3, traceability: without a retriever there is no source metadata to cite.
    traceability_doable = not corpus.traceability_required
    # Axis 4, access control: CAG cannot filter per user before the prompt.
    access_control_doable = not corpus.access_control_required

    viable_for_cag = all(
        [fits_in_window, cache_friendly, traceability_doable, access_control_doable]
    )
    if viable_for_cag:
        return "CAG"
    # A corpus that fits and is stable but fails traceability or access control
    # suits a hybrid: CAG for the stable part, RAG for the sensitive or citable part.
    if fits_in_window and cache_friendly:
        return "Hybrid"
    return "RAG"


# --- Proyecto 2, with defensible numbers ------------------------------------

PROYECTO_2 = CorpusProfile(
    name="Proyecto 2",
    estimated_tokens=250_000,  # budgets + transcripts + rate cards + addenda
    refresh_frequency_days=7,  # weekly commercial close
    traceability_required=True,  # legal requires citing the source
    access_control_required=True,  # confidential client information
)

CURRENT_MODEL = ModelProfile(
    name="gpt-4o-mini",
    context_window=128_000,
    cost_per_1k_input=0.00015,
    prefix_caching=True,
)


def _main(argv: list[str]) -> int:
    """CLI: ``python -m app.ingestion.architecture``."""
    corpus, model = PROYECTO_2, CURRENT_MODEL
    viability = assess_cag_viability(corpus, model)
    recommendation = recommend_architecture(corpus, model)

    print(
        f"Corpus '{corpus.name}': {corpus.estimated_tokens:,} tokens, "
        f"refresh every {corpus.refresh_frequency_days:g} days, "
        f"traceability={corpus.traceability_required}, "
        f"access_control={corpus.access_control_required}"
    )
    print(
        f"Model '{model.name}': window {model.context_window:,} tokens, "
        f"prefix_caching={model.prefix_caching}"
    )
    print()
    if viability.viable:
        print("CAG viability: viable (all 4 constraints green)")
    else:
        print("CAG viability: NOT viable")
        print(f"  Red constraints: {', '.join(viability.failing_constraints())}")
    print(f"Architecture recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
