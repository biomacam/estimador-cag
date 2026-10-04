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
)
from app.services.security import (
    frame_untrusted_input,
    new_untrusted_data_tag,
    untrusted_data_instructions,
)

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
    tag = new_untrusted_data_tag()
    framed_reference_projects = (
        [
            {
                "framed_name": frame_untrusted_input(rp.name, tag),
                "framed_summary": frame_untrusted_input(rp.summary, tag),
                "total_hours": rp.total_hours,
                "total_cost": rp.total_cost,
            }
            for rp in request.reference_projects
        ]
        if request.reference_projects
        else None
    )
    context = {
        "framed_description": frame_untrusted_input(request.description, tag),
        "untrusted_data_instructions": untrusted_data_instructions(
            tag, data_description="user-supplied data (project description and reference projects)"
        ),
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "reference_projects": framed_reference_projects,
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

