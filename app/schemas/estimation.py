from enum import Enum
from typing import Annotated, Literal

import litellm
from pydantic import AfterValidator, BaseModel, Field, model_validator

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


OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """One phase in a structured estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    """Structured estimation with arithmetic and low-confidence validation.

    Phases precede totals so a model can generate the breakdown before summing it.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(phase.cost_eur for phase in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"phases sum ({phase_sum} EUR) does not match total_cost_eur "
                f"({self.total_cost_eur} EUR); adjust either the phases or the total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                "description is too vague to size"
            )
        return self


class StructuredEstimationResponse(BaseModel):
    """Structured response contract without the compatibility Markdown field."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False


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
    """Validated result and locally rendered Markdown for existing clients."""

    text: str
    prompt_version: str
    result: EstimationResult
    cached: bool = False


class StreamEstimationRequest(BaseModel):
    """Streaming endpoint request � only the transcription, knobs are not exposed."""

    transcription: TranscriptionText
    model: str | None = Field(default=None, description="Override the default model")
    max_tokens: int = Field(default=4000, gt=0, le=16000)
