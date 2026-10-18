"""Presidio AnalyzerEngine factory configured for Spanish text.

Presidio's default configuration ships English spaCy models. On Hispanic names
("Laura Fernández", "Javier Romero") it silently returns *zero* PERSON entities,
a failure mode much worse than a false positive because nothing flags it. This
module is the single place that wires Presidio to the configured Spanish model.
"""

from __future__ import annotations

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.config import get_settings
from app.ingestion.pii.recognizers import BudgetIdRecognizer, ClientCodeRecognizer


@lru_cache
def build_analyzer() -> AnalyzerEngine:
    """Return a Presidio AnalyzerEngine wired to ``PRESIDIO_SPACY_MODEL``.

    A singleton because loading the spaCy model is slow and Presidio is
    thread-safe for read-only analysis. Custom recognizers are registered here so
    callers cannot forget them.
    """
    settings = get_settings()
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": settings.PRESIDIO_SPACY_MODEL}],
        }
    )
    engine = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["es"],
    )
    engine.registry.add_recognizer(BudgetIdRecognizer())
    engine.registry.add_recognizer(ClientCodeRecognizer())
    return engine
