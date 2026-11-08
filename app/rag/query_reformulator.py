"""Query understanding: turn a raw meeting transcript into a structured brief.

Embedding ~600 tokens of digression yields a vector that averages several
intents plus conversational noise. This stage distills the transcript into an
:class:`EstimationQuery` and composes a short technical English search string
aligned with the corpus. Both the primary extraction and the degraded fallback
go through ``LLMWrapper`` (Instructor + LiteLLM).
"""

from __future__ import annotations

import asyncio

import structlog

from app.config import get_settings
from app.rag.errors import ReformulationError
from app.rag.schemas import EstimationQuery

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a software-delivery analyst. Extract a structured project brief from "
    "a raw, messy client meeting transcript. Capture ONLY what the client wants to "
    "build and the constraints that bound it; ignore small talk, anecdotes and "
    "digressions. Normalise everything to concise technical English regardless of "
    "the transcript language. Leave a field empty/unknown when the transcript gives "
    "no evidence for it — never invent technologies, sectors or regulations."
)

_FALLBACK_SYSTEM_PROMPT = (
    "Rewrite the following client meeting transcript as a single short technical "
    "search query in English describing the software to build. One sentence, no "
    "preamble."
)


async def reformulate_query(transcript: str) -> EstimationQuery:
    """Distill a transcript into a structured :class:`EstimationQuery`.

    On the degraded fallback only ``function`` is set.

    Raises
    ------
    ReformulationError
        If both the structured extraction and the free-text fallback fail.
    """
    from app.dependencies import get_llm_wrapper

    settings = get_settings()
    wrapper = get_llm_wrapper()

    try:
        query, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_SYSTEM_PROMPT,
            user_message=transcript,
            response_model=EstimationQuery,
            model_override=settings.REFORMULATION_MODEL,
            # Reasoning tokens count against max_tokens; the 4000 default would truncate.
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
        return query
    except Exception as exc:  # noqa: BLE001 — degrade to a simpler rewrite.
        log.warning(
            "reformulation_fallback",
            reason="structured_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )

    try:
        result = await asyncio.to_thread(
            wrapper.complete,
            system_prompt=_FALLBACK_SYSTEM_PROMPT,
            user_message=transcript,
            model_override=settings.REFORMULATION_MODEL,
        )
        rewritten = (result.get("estimation") or "").strip()
        if not rewritten:
            raise ReformulationError("Fallback rewrite returned an empty query.")
        return EstimationQuery(function=rewritten)
    except ReformulationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReformulationError("Query reformulation failed.") from exc


def compose_search_text(query: EstimationQuery) -> str:
    """Compose the short technical string to embed; empty fields are dropped."""
    parts: list[str] = [query.function.strip()] if query.function.strip() else []

    if query.technologies:
        parts.append("with " + ", ".join(query.technologies))
    if query.sector:
        parts.append(f"for {query.sector}")
    if query.country:
        parts.append(f"in {query.country}")
    if query.regulations:
        parts.append(", ".join(query.regulations) + "-compliant")

    return " ".join(parts).strip()
