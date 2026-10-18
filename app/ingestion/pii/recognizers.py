"""Domain-specific Presidio recognizers.

Presidio detects PERSON, EMAIL, LOCATION, etc. out of the box, but the corpus
also carries:

* **BUDGET_ID**: ``BUDGET-YYYY-NNNN`` (year + four-digit serial).
* **CLIENT_CODE**: ``CLI-NNNN`` (four-digit serial).

Both are deterministic patterns, so a high-precision regex scores close to 1.0.
They are declared with ``supported_language="es"`` to activate alongside the
Spanish NLP pipeline configured in ``analyzer.build_analyzer``.
"""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


class BudgetIdRecognizer(PatternRecognizer):
    """Detects ``BUDGET-YYYY-NNNN`` budget identifiers."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="BUDGET_ID",
            name="BudgetIdRecognizer",
            patterns=[
                Pattern(
                    name="budget_id_pattern",
                    regex=r"\bBUDGET-\d{4}-\d{4}\b",
                    score=0.95,
                )
            ],
            supported_language="es",
        )


class ClientCodeRecognizer(PatternRecognizer):
    """Detects ``CLI-NNNN`` client codes."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CLIENT_CODE",
            name="ClientCodeRecognizer",
            patterns=[
                Pattern(
                    name="client_code_pattern",
                    regex=r"\bCLI-\d{4}\b",
                    score=0.95,
                )
            ],
            supported_language="es",
        )
