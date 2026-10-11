"""Critic service — independent audit of an ``EstimationResult``.

Stateless: (transcript, metadata, tier, estimation) → ``CriticFeedback``. It does not
mutate the session nor decide what to do with its output — that's the Boss's job.

A Critic failure never blocks the pipeline: the Boss receives a synthetic "accept
with zero confidence" verdict so the actor's estimation flows through unmodified.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.prompts.loader import render_critic_prompt
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import EstimationResult
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier

log = structlog.get_logger()

CRITIC_MAX_TOKENS = 1500
CRITIC_MAX_RETRIES = 3


class Critic:
    def __init__(self, *, llm_wrapper: Any, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: Tier,
        result: EstimationResult,
    ) -> CriticFeedback:
        system_prompt, user_message = render_critic_prompt(
            transcript=transcript, metadata=metadata, tier=tier, result=result
        )
        try:
            feedback, meta = self.llm_wrapper.complete_structured_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=CriticFeedback,
                model_override=self.model,
                max_tokens=CRITIC_MAX_TOKENS,
                max_retries=CRITIC_MAX_RETRIES,
            )
        except Exception as exc:  # noqa: BLE001 — degrade to accept, never fail the turn
            log.warning("critic_failed_fallback_accept", error_type=type(exc).__name__)
            return CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)

        log.info(
            "critic_completed",
            verdict=feedback.verdict,
            issue_count=len(feedback.issues),
            confidence=feedback.confidence_in_review,
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return feedback
