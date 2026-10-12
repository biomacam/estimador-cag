import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_every_prompt_version_tells_the_model_not_to_back_fit_to_a_stated_budget(version):
    request = EstimationRequest(
        description="A customer portal with invoices. Budget is 60k EUR for the first release.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system_prompt, _ = render_estimation_prompt(request, version=version)

    assert "never as the total" in system_prompt
    assert "exact sum" in system_prompt
