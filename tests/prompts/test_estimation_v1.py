"""Template-only tests for the v1 estimation prompt: verify what render_estimation_prompt
produces, never call an LLM. Must stay fast — no network, no provider mocking needed.
"""

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)


def _request(
    *,
    description: str = "Build a booking system for a yoga studio with class scheduling.",
    project_type: ProjectType = ProjectType.WEB_SAAS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.LINE_ITEMS,
    reference_projects: list[ReferenceProject] | None = None,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        reference_projects=reference_projects,
    )


def test_user_prompt_contains_the_literal_description() -> None:
    description = "Build a booking system for a yoga studio with class scheduling and payments."
    _, user = render_estimation_prompt(_request(description=description))

    assert description in user


def test_system_prompt_output_format_instruction_matches_only_the_selected_format() -> None:
    phases_system, _ = render_estimation_prompt(_request(output_format=OutputFormat.PHASES_TABLE))
    narrative_system, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))

    assert "Format the output as a Markdown table of phases" in phases_system
    assert "Format the output as a Markdown table of phases" not in narrative_system

    assert "Format the output as a narrative description in prose" in narrative_system
    assert "Format the output as a narrative description in prose" not in phases_system


def test_reference_projects_are_looped_into_the_system_prompt_only_when_present() -> None:
    system_without, _ = render_estimation_prompt(_request())
    assert "similar past projects" not in system_without.lower()

    reference_projects = [
        ReferenceProject(
            name="Fitness CRM", summary="Booking and payments for a gym chain.",
            total_hours=120, total_cost=7500,
        ),
        ReferenceProject(
            name="Spa scheduler", summary="Appointment booking for a spa.",
            total_hours=80, total_cost=5000,
        ),
    ]
    system_with, _ = render_estimation_prompt(_request(reference_projects=reference_projects))

    assert "Fitness CRM" in system_with and "Booking and payments for a gym chain." in system_with
    assert "Spa scheduler" in system_with and "Appointment booking for a spa." in system_with
    assert "120" in system_with and "7500" in system_with


def test_system_prompt_lists_assumptions_per_phase_only_when_detailed() -> None:
    detailed_system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
    summary_system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.SUMMARY))

    assert "assumptions that justify its scope" in detailed_system
    assert "assumptions that justify its scope" not in summary_system

    assert "Keep the estimation brief" in summary_system
    assert "Keep the estimation brief" not in detailed_system
