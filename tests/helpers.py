"""Shared test helpers for asserting the untrusted-data delimiter contract
(see app.services.llm_service.frame_untrusted_input) without hardcoding the
tag in assertions about the wrapped text.
"""

import re
from app.schemas.estimation import EstimationResult


def valid_estimation_result() -> EstimationResult:
    return EstimationResult(
        summary="Build a booking system with payments.",
        confidence_pct=80,
        phases=[{
            "name": "Implementation", "duration_weeks": 4, "cost_eur": 10000,
            "summary": "Build booking and payment integrations.",
        }],
        total_duration_weeks=4,
        total_cost_eur=10000,
    )

_FRAMED_RE = re.compile(r"^<user-data-([a-z0-9]+)>\n(.*)\n</user-data-\1>$", re.DOTALL)


def unwrap_untrusted_input(framed: str) -> str:
    """Strip the ``<user-data-{tag}>...</user-data-{tag}>`` wrapper and return the
    original text. Raises AssertionError if ``framed`` doesn't match that shape.
    """
    match = _FRAMED_RE.fullmatch(framed)
    assert match is not None, f"Expected a <user-data-...> framed string, got: {framed!r}"
    return match.group(2)
