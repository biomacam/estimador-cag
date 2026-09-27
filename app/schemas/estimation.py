from enum import Enum
from typing import Annotated, Literal

import litellm
from pydantic import AfterValidator, BaseModel, Field

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]
MAX_TRANSCRIPTION_TOKENS = 50_000


def _validate_transcription_token_limit(transcription: str) -> str:
    token_count = litellm.token_counter(model="gpt-4o-mini", text=transcription)
    if token_count > MAX_TRANSCRIPTION_TOKENS:
        raise ValueError(
            f"Transcription must not exceed {MAX_TRANSCRIPTION_TOKENS:,} tokens"
        )
    return transcription


TranscriptionText = Annotated[
    str,
    Field(min_length=50, description="Meeting transcription text (maximum 50,000 tokens)"),
    AfterValidator(_validate_transcription_token_limit),
]


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """A similar past project the caller supplies as extra grounding context."""

    name: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=1000)
    total_hours: int = Field(gt=0, le=100_000)
    total_cost: float = Field(gt=0, le=10_000_000)


class EstimationRequest(BaseModel):
    """Typed contract for the form-based estimation endpoint."""

    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = Field(
        default=None,
        max_length=5,
        description="Optional similar past projects to ground the estimate",
    )


class TokenUsage(BaseModel):
    """Token consumption details from the LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]


class EstimationResponse(BaseModel):
    """Response containing the generated free-form estimation text."""

    text: str
    prompt_version: str


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request � only the transcription, knobs are not exposed."""

    transcription: TranscriptionText
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
