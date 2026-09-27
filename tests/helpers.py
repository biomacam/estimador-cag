"""Shared test helpers for asserting the untrusted-data delimiter contract
(see app.services.llm_service.frame_untrusted_input) without hardcoding the
per-request random tag anywhere in the test suite.
"""

import re

_FRAMED_RE = re.compile(r"^<user-data-([0-9a-f]+)>\n(.*)\n</user-data-\1>$", re.DOTALL)


def unwrap_untrusted_input(framed: str) -> str:
    """Strip the ``<user-data-{tag}>...</user-data-{tag}>`` wrapper and return the
    original text. Raises AssertionError if ``framed`` doesn't match that shape.
    """
    match = _FRAMED_RE.fullmatch(framed)
    assert match is not None, f"Expected a <user-data-...> framed string, got: {framed!r}"
    return match.group(2)
