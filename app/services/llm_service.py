"""Estimation orchestration: prompt building, optional preprocessing, and dispatch
to the LLM. The actual provider calls now live in :mod:`app.services.llm_wrapper`,
so this module focuses on Session 2 concerns (knobs, prompt assembly) while the
wrapper handles cache, fallback, and cost tracking transparently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from instructor.core import IncompleteOutputException, InstructorRetryException
import openai
import structlog
from litellm.exceptions import APIError as LLMProviderAPIError
from pydantic import ValidationError

if TYPE_CHECKING:
    from app.cache.semantic import EstimationSemanticCache

from app.context.examples import format_examples_for_prompt, select_examples
from app.dependencies import get_llm_wrapper
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts.loader import render_conversational_prompt, render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel, EstimationRequest, EstimationResponse, EstimationResult, ExampleFormat, OutputFormat,
    PreprocessingMode, ProjectType,
)
from app.services.evaluation import OK_FINISH_REASONS
from app.services.llm_errors import PROVIDER_ERROR, LLMFailureReason, classify_llm_error
from app.services.llm_wrapper import LLMTruncatedResponseError
from app.services.metadata_extractor import update_metadata
from app.services.security import (
    frame_untrusted_input,
    new_untrusted_data_tag,
    untrusted_data_instructions,
)
from app.sessions import Session

log = structlog.get_logger()

DEFAULT_MAX_TOKENS = 4000
EXTRACTION_MAX_TOKENS = 1500

# Fixed, safe-to-expose message: provider exceptions may embed API key fragments,
# org ids or account URLs in their str() — those details are logged, never returned.
LLM_PROVIDER_ERROR_MESSAGE = "The LLM provider failed to generate the estimation."


class LLMServiceError(Exception):
    """Raised when the LLM provider fails or returns a response we must not use as-is.

    ``reason`` is the safe summary exposed to clients next to the generic message.
    """

    def __init__(
        self, message: str = LLM_PROVIDER_ERROR_MESSAGE, *, reason: LLMFailureReason = PROVIDER_ERROR
    ) -> None:
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Prompt building blocks
#
# The two ACTIVE_OUTPUT_PROMPT variants live side by side so the instructor
# can switch between them in the live session (Block 3.4) by editing the
# ACTIVE_OUTPUT_PROMPT assignment below. Uvicorn `--reload` picks up the
# change automatically.
# ---------------------------------------------------------------------------

PROMPT_OUTPUT_BASIC = "Generate an estimation for the project described above."

PROMPT_OUTPUT_STRUCTURED = """\
Generate the estimation with this exact structure:

## Project summary
[2-3 sentences describing the project scope and goals]

## Task breakdown
| Task | Hours | Cost (EUR) |
[one row per task; cost = hours * 62.50 EUR for developer tasks]

## Totals
- Total hours: [number]
- Total cost: [number] EUR
- Recommended team: [composition]
- Estimated duration: [weeks]

## Risks and assumptions
- [3-5 bullet points covering technical risks, scope assumptions, and external dependencies]
"""

# >>> Block 3.4 live switch: change the right-hand side to PROMPT_OUTPUT_STRUCTURED
ACTIVE_OUTPUT_PROMPT = PROMPT_OUTPUT_BASIC


INLINE_CLEANING_BLOCK = """\
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation."""


EXTRACTION_SYSTEM_PROMPT = (
    "You are an analyst. Read the meeting transcription and produce a clean, "
    "deduplicated bullet list of functional requirements, non-functional "
    "requirements, integrations, constraints and explicit deadlines. Ignore "
    "fillers, divagations and off-topic remarks. Output Markdown only."
)


# Prompt injection defense helpers (new_untrusted_data_tag, frame_untrusted_input,
# untrusted_data_instructions) now live in app.services.security so both this
# module and app.prompts.loader can share them without a circular import.


def _untrusted_data_instructions(tag: str) -> str:
    return untrusted_data_instructions(tag, data_description="meeting transcription")


@dataclass
class GenerationOptions:
    """Per-request knobs that drive prompt construction and the LLM call."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 3,
    use_examples: bool = True,
    inline_cleaning: bool = False,
    untrusted_data_tag: str | None = None,
) -> str:
    """Assemble the system prompt with role, rates, output spec and (optionally) examples."""
    role = (
        "You are a senior software consultant with 15+ years of experience in project "
        "estimation. Your task is to produce a detailed software project estimation based "
        "on a meeting transcription provided by the user."
    )
    rates = (
        "Use a developer rate of approximately 73.50 EUR/hour (588 EUR/day) and a designer "
        "rate of approximately 50 EUR/hour (400 EUR/day). Provide realistic, well-justified "
        "numbers."
    )

    examples_block = ""
    if use_examples and num_examples > 0:
        rendered = format_examples_for_prompt(select_examples(num_examples), example_format)
        if rendered:
            examples_block = (
                "Below are reference estimations from previous projects. Use them as a guide "
                "for structure, level of detail, and realistic pricing. Adapt the content to "
                "match the specific project described in the transcription.\n\n"
                + rendered
            )

    cleaning_block = INLINE_CLEANING_BLOCK if inline_cleaning else ""
    untrusted_data_block = (
        _untrusted_data_instructions(untrusted_data_tag) if untrusted_data_tag else ""
    )

    sections = [
        role,
        cleaning_block,
        rates,
        ACTIVE_OUTPUT_PROMPT,
        untrusted_data_block,
        examples_block,
    ]
    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# LLM dispatch (single seam; tests monkeypatch this)
# ---------------------------------------------------------------------------


def _invoke_llm(
    *,
    system_prompt: str,
    user_message: str,
    model_override: str | None,
    max_tokens: int,
    thinking_budget: int | None,
) -> dict[str, Any]:
    """Single seam through which every LLM call passes. Tests monkeypatch this."""
    wrapper = get_llm_wrapper()
    return wrapper.complete(
        system_prompt=system_prompt,
        user_message=user_message,
        model_override=model_override,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )


# ---------------------------------------------------------------------------
# Two-phase preprocessing (phase 1: requirement extraction)
# ---------------------------------------------------------------------------


def extract_requirements(
    transcription: str,
    opts: GenerationOptions,
    untrusted_data_tag: str,
) -> tuple[str, dict, float]:
    """Run the cheap phase-1 LLM call that turns a raw transcription into clean requirements.

    Returns ``(requirements_text, usage_dict, cost_usd)``.
    """
    log.info("extracting_requirements", model_override=opts.model)

    try:
        result = _invoke_llm(
            system_prompt=EXTRACTION_SYSTEM_PROMPT
            + "\n\n"
            + _untrusted_data_instructions(untrusted_data_tag),
            user_message=frame_untrusted_input(transcription, untrusted_data_tag),
            model_override=opts.model,
            max_tokens=EXTRACTION_MAX_TOKENS,
            thinking_budget=None,
        )
    except LLMProviderAPIError as exc:
        log.error("requirements_extraction_failed", error=str(exc), error_type=type(exc).__name__)
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE) from exc
    except LLMTruncatedResponseError as exc:
        log.error("requirements_extraction_truncated", error=str(exc))
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE) from exc

    return (
        result["estimation"],
        {
            "input": result["usage"]["input_tokens"],
            "output": result["usage"]["output_tokens"],
        },
        float(result.get("cost_usd", 0.0)),
    )


# ---------------------------------------------------------------------------
# Typed entrypoint (Jinja2-rendered prompts, versioned)
# ---------------------------------------------------------------------------


def _invoke_structured_llm(
    *, system_prompt: str, user_message: str,
) -> tuple[EstimationResult, dict[str, Any]]:
    return get_llm_wrapper().complete_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=EstimationResult,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def _invoke_structured_chat(
    *, messages: list[dict[str, str]]
) -> tuple[EstimationResult, dict[str, Any]]:
    return get_llm_wrapper().complete_structured_chat(
        messages=messages,
        response_model=EstimationResult,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def render_structured_estimation(result: EstimationResult, output_format: OutputFormat) -> str:
    """Render validated data locally for clients that still display Markdown."""
    def inline(text: str) -> str:
        return " ".join(text.split()).replace("|", "\\|")

    lines = ["## Project summary", result.summary, "", "## Phases"]
    if output_format == OutputFormat.PHASES_TABLE:
        lines.extend(["| Phase | Weeks | Cost (EUR) | Summary |", "| --- | ---: | ---: | --- |"])
        lines.extend(
            f"| {inline(phase.name)} | {phase.duration_weeks} | {phase.cost_eur:,} | {inline(phase.summary)} |"
            for phase in result.phases
        )
    elif output_format == OutputFormat.LINE_ITEMS:
        lines.extend(
            f"- {inline(phase.name)}: {phase.duration_weeks} weeks, {phase.cost_eur:,} EUR. {inline(phase.summary)}"
            for phase in result.phases
        )
    else:
        lines.extend(
            f"{phase.name}: {phase.summary} Duration: {phase.duration_weeks} weeks. Cost: {phase.cost_eur:,} EUR.\n"
            for phase in result.phases
        )
    lines.extend([
        "", "## Totals",
        f"- Total cost: {result.total_cost_eur:,} EUR",
        f"- Estimated duration: {result.total_duration_weeks} weeks",
        f"- Confidence: {result.confidence_pct}%",
    ])
    return "\n".join(lines)


def generate_typed_estimation(request: EstimationRequest, version: str = "v1") -> dict[str, Any]:
    """Generate validated structured data and render compatibility text locally."""
    # The structured-output contract and out-of-scope instructions live in
    # app/prompts/estimation/_shared/output_schema.j2, included by every version.
    system_prompt, user_message = render_estimation_prompt(request, version=version)

    log.info(
        "generating_typed_estimation",
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        prompt_version=version,
    )

    return _run_structured_estimation(
        lambda: _invoke_structured_llm(system_prompt=system_prompt, user_message=user_message),
        request.output_format,
    )


_ESTIMATION_FAILURES = (
    LLMProviderAPIError,
    openai.APIError,
    LLMTruncatedResponseError,
    IncompleteOutputException,
    InstructorRetryException,
    ValidationError,
)


def _run_structured_estimation(
    invoke: Callable[[], tuple[EstimationResult, dict[str, Any]]], output_format: OutputFormat
) -> dict[str, Any]:
    try:
        result, meta = invoke()
        result = EstimationResult.model_validate(result.model_dump())
        # Third layer of scope robustness: normalises the rare edge case where
        # the prompt instructions and the schema validator both allowed a
        # low-confidence answer through without the "Out of scope:" prefix.
        result = enforce_scope_response(result)
    except _ESTIMATION_FAILURES as exc:
        reason = classify_llm_error(exc)
        log.error(
            "llm_estimation_failed",
            reason_code=reason.code,
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE, reason=reason) from exc

    return {
        "estimation": render_structured_estimation(result, output_format),
        "result": result,
        "cache_hit": meta.get("cache_hit", False),
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


class EstimationService:
    """HTTP-independent entry point for the typed estimation pipeline."""

    def __init__(
        self,
        *,
        openai_client: Any | None = None,
        semantic_cache: "EstimationSemanticCache | None" = None,
    ) -> None:
        self.openai_client = openai_client
        self.semantic_cache = semantic_cache

    def estimate(self, request: EstimationRequest, version: str = "v1") -> EstimationResponse:
        # Input guardrails run before any cache lookup or LLM call — a rejected
        # description must never be served from cache nor reach the provider.
        check_input(request.description, openai_client=self.openai_client)

        if self.semantic_cache is not None:
            semantic_hit = self.semantic_cache.lookup(request, version)
            if semantic_hit is not None:
                log.info("estimation_cache_hit", kind="semantic")
                return EstimationResponse(
                    text=render_structured_estimation(semantic_hit, request.output_format),
                    prompt_version=version,
                    result=semantic_hit,
                    cached=True,
                )

        generated = generate_typed_estimation(request, version=version)
        if self.semantic_cache is not None and not generated["cache_hit"]:
            self.semantic_cache.store(request, generated["result"], version)

        return EstimationResponse(
            text=generated["estimation"],
            prompt_version=version,
            result=generated["result"],
            cached=generated["cache_hit"],
        )

    def estimate_conversational(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        version: str = "v1",
    ) -> EstimationResponse:
        """One turn of a session: ``transcript`` already includes the attachments' text.

        The system prompt is re-rendered each turn from the session's ``ProjectMetadata``
        and kept as the history's system prompt, so the window never drops it. No cache
        is used: the answer depends on the history, not just on this transcript.
        """
        check_input(transcript, openai_client=self.openai_client)

        system_prompt, user_message = render_conversational_prompt(
            transcript,
            metadata=session.metadata,
            project_type=project_type.value,
            detail_level=detail_level.value,
            output_format=output_format.value,
            version=version,
        )
        messages = [
            *session.history.to_messages_list(system_prompt),
            {"role": "user", "content": user_message},
        ]
        log.info(
            "generating_conversational_estimation",
            session_id=session.session_id,
            history_messages=len(session.history.messages),
            metadata_is_empty=session.metadata.is_empty(),
            transcript_chars=len(transcript),
            prompt_version=version,
        )

        generated = _run_structured_estimation(
            lambda: _invoke_structured_chat(messages=messages), output_format
        )
        result: EstimationResult = generated["result"]

        session.history.add("user", user_message)
        session.history.add("assistant", result.model_dump_json())
        session.metadata = update_metadata(
            previous=session.metadata, transcript=transcript, result=result
        )

        return EstimationResponse(
            text=generated["estimation"],
            prompt_version=version,
            result=result,
            cached=False,
        )


def generate_estimation(
    transcription: str,
    opts: GenerationOptions | None = None,
) -> dict[str, Any]:
    """Generate a software estimation from a meeting transcription using the configured LLM."""
    opts = opts or GenerationOptions()

    t0 = time.perf_counter()
    untrusted_data_tag = new_untrusted_data_tag()

    prep_usage = {"input": 0, "output": 0}
    prep_cost = 0.0
    extracted_requirements: str | None = None
    user_input = transcription

    if opts.preprocessing == "two_phase":
        extracted_requirements, prep_usage, prep_cost = extract_requirements(
            transcription, opts, untrusted_data_tag
        )
        user_input = extracted_requirements

    system_prompt = build_system_prompt(
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        inline_cleaning=(opts.preprocessing == "inline_cleaning"),
        untrusted_data_tag=untrusted_data_tag,
    )

    log.info(
        "generating_estimation",
        model_override=opts.model,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
    )

    try:
        result = _invoke_llm(
            system_prompt=system_prompt,
            user_message=frame_untrusted_input(user_input, untrusted_data_tag),
            model_override=opts.model,
            max_tokens=opts.max_tokens,
            thinking_budget=opts.thinking_budget,
        )
    except LLMProviderAPIError as exc:
        log.error("llm_call_failed", error=str(exc), error_type=type(exc).__name__)
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE) from exc
    except LLMTruncatedResponseError as exc:
        log.error("llm_call_truncated", error=str(exc))
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE) from exc

    if result["finish_reason"] not in OK_FINISH_REASONS:
        log.error("llm_response_truncated", finish_reason=result["finish_reason"])
        raise LLMServiceError(LLM_PROVIDER_ERROR_MESSAGE)

    result["usage"]["preprocessing_input_tokens"] = prep_usage["input"]
    result["usage"]["preprocessing_output_tokens"] = prep_usage["output"]
    result["preprocessing"] = opts.preprocessing
    result["extracted_requirements"] = extracted_requirements
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["cost_usd"] = round(float(result.get("cost_usd", 0.0)) + prep_cost, 6)
    # ``cache_hit`` is whatever the wrapper returned for the main estimation call.
    result.setdefault("cache_hit", False)

    return result
