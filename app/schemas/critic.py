"""Structured feedback schema produced by the Critic.

The schema is the contract between Critic and Boss: the Boss reads each issue's
``category``, ``severity`` and ``field_path`` to decide whether to accept the
actor's output, iterate with feedback, or synthesise a fallback. Free-text
reviews would force the Boss to parse prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]

CriticIssueSeverity = Literal["critical", "major", "minor"]

CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    """A single defect. ``field_path`` points at the estimation field (``phases[2].cost_eur``)."""

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    """Top-level Critic output; the Boss treats it as authoritative."""

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def iteration_requires_blocking_issue(self) -> CriticFeedback:
        if self.verdict == "needs_iteration" and not any(
            issue.severity in {"critical", "major"} for issue in self.issues
        ):
            raise ValueError(
                "verdict 'needs_iteration' requires at least one critical or major issue; "
                "minor-only issues should accept"
            )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why")
        return self
