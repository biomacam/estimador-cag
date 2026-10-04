"""Second LLM call per session turn: extract ``ProjectMetadata`` from the latest exchange."""

from __future__ import annotations

from typing import Any

import structlog

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimation import EstimationResult
from app.sessions import ProjectMetadata

log = structlog.get_logger()

EXTRACTION_MAX_TOKENS = 1000
EXTRACTION_MAX_RETRIES = 2


def _invoke_extractor(messages: list[dict[str, str]]) -> tuple[ProjectMetadata, dict[str, Any]]:
    return get_llm_wrapper().complete_structured_chat(
        messages=messages,
        response_model=ProjectMetadata,
        model_override=get_settings().METADATA_EXTRACTOR_MODEL,
        max_tokens=EXTRACTION_MAX_TOKENS,
        max_retries=EXTRACTION_MAX_RETRIES,
    )


def update_metadata(
    *, previous: ProjectMetadata, transcript: str, result: EstimationResult
) -> ProjectMetadata:
    """Return ``previous`` merged with the facts extracted from this turn.

    Any failure returns ``previous`` unchanged: losing one refresh must not break the turn.
    """
    system_prompt, user_message = render_metadata_extraction_prompt(
        transcript=transcript, result=result, previous=previous
    )
    try:
        extracted, meta = _invoke_extractor(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
    except Exception as exc:  # noqa: BLE001 — provider, validation and network errors alike
        log.warning("metadata_extraction_failed", error_type=type(exc).__name__)
        return previous

    merged = previous.merge_with(extracted)
    log.info(
        "metadata_extraction_completed",
        latency_ms=meta.get("latency_ms"),
        technologies=len(merged.mentioned_technologies),
    )
    return merged
