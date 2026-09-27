"""Template-only tests for the v2 estimation prompt: verify it's a deliberately
different variant from v1 (tone + examples), while keeping the same security
boundary. Never calls an LLM — must stay fast.
"""

from app.prompts.loader import PromptVersionNotFoundError, render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

import pytest


def _request(
    *,
    description: str = "Build a booking system for a yoga studio with class scheduling.",
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.LINE_ITEMS,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_v2_uses_a_different_tone_and_examples_than_v1() -> None:
    system_v1, _ = render_estimation_prompt(_request(), version="v1")
    system_v2, _ = render_estimation_prompt(_request(), version="v2")

    assert system_v1 != system_v2
    assert "pragmatic engineering lead" in system_v2
    assert "pragmatic engineering lead" not in system_v1
    # v2 ships its own few-shot set, not v1's.
    assert "E-commerce marketplace" in system_v2
    assert "E-commerce marketplace" not in system_v1


def test_v2_user_prompt_still_contains_the_literal_description() -> None:
    description = "Build a booking system for a yoga studio with class scheduling and payments."
    _, user = render_estimation_prompt(_request(description=description), version="v2")

    assert description in user


def test_unknown_prompt_version_raises_a_clear_error() -> None:
    with pytest.raises(PromptVersionNotFoundError):
        render_estimation_prompt(_request(), version="v99")
