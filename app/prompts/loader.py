"""Jinja2 loader for versioned estimation prompts.

Prompts live under ``app/prompts/estimation/<version>/`` as ``.j2`` templates so
they can be edited and versioned independently of the Python code that calls
them. ``render_estimation_prompt`` is the only entry point other modules need.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from app.schemas.estimation import (
    OUT_OF_SCOPE_PREFIX,
    LOW_CONFIDENCE_THRESHOLD,
    EstimationRequest,
    EstimationResult,
    ReferenceProject,
)
from app.services.security import (
    frame_untrusted_input,
    new_untrusted_data_tag,
    untrusted_data_instructions,
)
from app.sessions import ProjectMetadata

PROMPTS_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class PromptVersionNotFoundError(ValueError):
    """Raised when ``render_estimation_prompt`` is asked for a version with no templates."""


def render_estimation_prompt(
    request: EstimationRequest, version: str = "v1"
) -> tuple[str, str]:
    """Render the (system, user) messages for the estimation prompt at ``version``.

    ``request.description`` and each ``reference_projects[i].{name,summary}`` are
    untrusted external input, so none of them reach the templates as a raw string:
    they're all wrapped in the *same* fixed ``<user-data-{tag}>``
    delimiter (same boundary used by the transcription-based flow in
    ``llm_service.py``) and the system prompt is told to treat any such block as
    data, never as instructions, wherever it appears.
    """
    return _render(
        description=request.description,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        reference_projects=request.reference_projects,
        metadata=None,
        version=version,
    )


def render_conversational_prompt(
    transcript: str,
    *,
    metadata: ProjectMetadata,
    project_type: str,
    detail_level: str,
    output_format: str,
    version: str = "v1",
) -> tuple[str, str]:
    """(system, user) for a session turn: ``transcript`` already includes attachment text.

    Not bound by ``EstimationRequest.description``'s 2000-character limit. The system
    prompt carries the session's ``ProjectMetadata``; the history is added by the caller.
    """
    return _render(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        reference_projects=None,
        metadata=metadata,
        version=version,
    )


def render_metadata_extraction_prompt(
    *,
    transcript: str,
    result: EstimationResult,
    previous: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """(system, user) for the per-turn call that extracts ``ProjectMetadata``."""
    tag = new_untrusted_data_tag()
    context = {
        "framed_transcript": frame_untrusted_input(transcript, tag),
        "untrusted_data_instructions": untrusted_data_instructions(
            tag, data_description="user-supplied data (the conversation transcript and project facts)"
        ),
        "result": result,
        "previous": _frame_metadata(previous, tag),
    }
    try:
        system = _env.get_template(f"metadata_extraction/{version}/system.j2").render(context)
        user = _env.get_template(f"metadata_extraction/{version}/user.j2").render(context)
    except TemplateNotFound as exc:
        raise PromptVersionNotFoundError(f"Unknown prompt version: {version!r}") from exc
    return system, user


def _frame_metadata(metadata: ProjectMetadata | None, tag: str) -> dict[str, object] | None:
    """Metadata is LLM-extracted from user text, so its free text is framed as data too."""
    if metadata is None or metadata.is_empty():
        return None
    return {
        "project_name": (
            frame_untrusted_input(metadata.project_name, tag) if metadata.project_name else "unknown"
        ),
        "assumed_team_size": metadata.assumed_team_size or "unknown",
        "mentioned_technologies": (
            frame_untrusted_input(", ".join(metadata.mentioned_technologies), tag)
            if metadata.mentioned_technologies
            else "none"
        ),
        "agreed_scope": (
            frame_untrusted_input(metadata.agreed_scope, tag)
            if metadata.agreed_scope
            else "not yet agreed"
        ),
    }


def _render(
    *,
    description: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    reference_projects: list[ReferenceProject] | None,
    metadata: ProjectMetadata | None,
    version: str,
) -> tuple[str, str]:
    tag = new_untrusted_data_tag()
    framed_reference_projects = (
        [
            {
                "framed_name": frame_untrusted_input(rp.name, tag),
                "framed_summary": frame_untrusted_input(rp.summary, tag),
                "total_hours": rp.total_hours,
                "total_cost": rp.total_cost,
            }
            for rp in reference_projects
        ]
        if reference_projects
        else None
    )
    context = {
        "framed_description": frame_untrusted_input(description, tag),
        "untrusted_data_instructions": untrusted_data_instructions(
            tag, data_description="user-supplied data (project description and reference projects)"
        ),
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "reference_projects": framed_reference_projects,
        "conversational": metadata is not None,
        "framed_metadata": _frame_metadata(metadata, tag),
        # Sourced from the schema so the prompt can never drift from what the
        # EstimationResult model_validators actually enforce.
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "out_of_scope_prefix": OUT_OF_SCOPE_PREFIX,
    }
    try:
        system = _env.get_template(f"estimation/{version}/system.j2").render(context)
        user = _env.get_template(f"estimation/{version}/user.j2").render(context)
    except TemplateNotFound as exc:
        raise PromptVersionNotFoundError(f"Unknown prompt version: {version!r}") from exc
    return system, user

