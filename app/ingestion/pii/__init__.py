"""GDPR-oriented pseudonymization layer.

Three concerns, three modules:

* ``analyzer``: initializes Presidio's ``AnalyzerEngine`` against a Spanish spaCy
  model. Without an explicit Spanish model Presidio defaults to English and
  silently misses Hispanic names.

* ``recognizers``: custom :class:`PatternRecognizer` classes for the project's
  domain-specific identifiers (``BUDGET-YYYY-NNNN`` and ``CLI-NNNN``).

* ``pseudonymizer``: the :class:`ConsistentPseudonymizer`. The same original
  value always maps to the same pseudonym, across documents and runs, backed by a
  persistent :class:`MappingStore` (Art. 17 "right to be forgotten").
"""

from app.ingestion.pii.analyzer import build_analyzer
from app.ingestion.pii.mapping_store import (
    InMemoryMappingStore,
    MappingStore,
    PostgresMappingStore,
)
from app.ingestion.pii.pseudonymizer import (
    ConsistentPseudonymizer,
    PseudonymizationResult,
)
from app.ingestion.pii.recognizers import (
    BudgetIdRecognizer,
    ClientCodeRecognizer,
)

__all__ = [
    "BudgetIdRecognizer",
    "ClientCodeRecognizer",
    "ConsistentPseudonymizer",
    "InMemoryMappingStore",
    "MappingStore",
    "PostgresMappingStore",
    "PseudonymizationResult",
    "build_analyzer",
]
