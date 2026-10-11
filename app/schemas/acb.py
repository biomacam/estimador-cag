"""DTOs for the Actor-Critic-Boss audit trail, shared by the Boss and ``ACBResponse``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BossDecision = Literal["accept", "iterate", "synthesize"]


class ACBIteration(BaseModel):
    """Audit record for one actor + critic round."""

    iteration: int = Field(ge=0)
    decision_after: BossDecision
    critic_verdict: str
    critic_confidence: int = Field(ge=0, le=100)
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int = Field(ge=0)
