"""Cumulative summarizer for evicted conversation turns.

Invoked by ``CompressionPolicy`` whenever non-anchor turns fall off the sliding
window. It takes the previous running summary (may be empty) plus the freshly
evicted messages and folds them into a new summary that replaces the previous
one — a single rolling summary per session, not a chain.

On failure the previous summary is kept intact: better to lose one compaction
pass than to wipe state the model has already committed to.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_conversation_summary_prompt
from app.sessions.models import Message

log = structlog.get_logger()


class _SummaryEnvelope(BaseModel):
    """Single-field wrapper so Instructor can validate the running summary."""

    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, *, llm_wrapper: Any, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str | None:
        """Return the updated cumulative summary, or ``previous_summary`` on any error."""
        if not evicted:
            return previous_summary

        system_prompt, user_message = render_conversation_summary_prompt(
            previous_summary=previous_summary, evicted=evicted
        )
        try:
            envelope, meta = self.llm_wrapper.complete_structured_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_SummaryEnvelope,
                model_override=self.model,
                max_tokens=1000,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 — compression is best-effort
            log.warning("summarizer_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            return previous_summary

        log.info(
            "summarizer_completed",
            evicted_count=len(evicted),
            previous_chars=len(previous_summary or ""),
            new_chars=len(envelope.summary),
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return envelope.summary
