"""Schemas for the hand-written agent.

Three families of models live here:

* **Tool argument models** (``SearchBudgetsArgs`` …) — every
  ``json.loads(function_call.arguments)`` is validated into one of these BEFORE the
  tool runs, so a malformed or hallucinated argument raises a
  ``pydantic.ValidationError`` the loop hands back to the model as an error string
  to self-correct from.
* **Trace models** (``AgentStep`` / ``AgentTrace``) — the reasoning→action→
  observation record of a run. ``AgentTrace.render`` prints the ``STEP N`` console
  format.
* **Result models** (``AgentComponent`` / ``AgentEstimate`` / ``AgentRunResult``)
  — a deliberately LIGHT final estimate, distinct from the heavy RAG ``Estimate``.
  The terminal ``responses.parse`` call in the loop fills ``AgentEstimate``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]

# A model may occasionally emit NUL or other control characters inside a malformed
# unicode escape in a tool argument; strip them so they never poison the trace.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class SearchBudgetsFilters(BaseModel):
    """Optional structural filters for a budget search.

    Mirrors the nullable ``filters`` object in the tool schema. ``None`` on a field
    means "do not constrain on this axis".
    """

    sectors: list[str] | None = Field(
        default=None,
        description="Restrict to these client sectors (e.g. ['logistics', 'industrial']).",
    )
    component_type: str | None = Field(
        default=None,
        description="Free-text hint about the kind of component (e.g. 'mobile app').",
    )


class SearchBudgetsArgs(BaseModel):
    """Validated arguments for the ``search_budgets`` tool."""

    query: str = Field(min_length=1)
    filters: SearchBudgetsFilters | None = None


class ComponentInput(BaseModel):
    """One component the agent wants costed, with its historical references."""

    name: str = Field(min_length=1)
    reference_amounts: list[float] = Field(
        description="Historical amounts (engineer-hours) for analogous work, from search_budgets."
    )


class CalculateEstimateArgs(BaseModel):
    """Validated arguments for the ``calculate_estimate`` tool."""

    components: list[ComponentInput]


class ValidateComponentInput(BaseModel):
    """One line of the estimate to validate."""

    name: str = Field(min_length=1)
    estimated_hours: float
    reference_amounts: list[float] = Field(default_factory=list)


class ValidateEstimateArgs(BaseModel):
    """Validated arguments for the ``validate_estimate`` tool."""

    components: list[ValidateComponentInput]
    total_hours: float


class AgentStep(BaseModel):
    """One reason→act→observe step of the loop."""

    step: int = Field(ge=1)
    reasoning_summary: str | None = Field(
        default=None,
        description="Model reasoning summary for this step (Responses API reasoning summary).",
    )
    tool: str = Field(description="Name of the invoked tool.")
    tool_args: dict[str, Any] = Field(
        description="Arguments the model passed to the tool."
    )
    observation: str = Field(description="Human-readable summary of the tool result.")


class AgentTrace(BaseModel):
    """Ordered record of everything the agent did, for auditing."""

    steps: list[AgentStep] = Field(default_factory=list)

    def render(self) -> str:
        """Render the trace in the ``STEP N`` console format."""
        if not self.steps:
            return "(no tool steps — the agent answered without calling any tool)"
        blocks: list[str] = []
        for s in self.steps:
            reasoning = s.reasoning_summary or "(no reasoning summary emitted)"
            args = _CONTROL_CHARS.sub(
                "", json.dumps(s.tool_args, ensure_ascii=False, default=str)
            )
            blocks.append(
                f"STEP {s.step}\n"
                f"  reasoning:   {reasoning}\n"
                f"  action:      {s.tool}({args})\n"
                f"  observation: {s.observation}"
            )
        return "\n\n".join(blocks)


class AgentComponent(BaseModel):
    """One costed component in the final estimate."""

    name: str
    estimated_hours: float = Field(ge=0)
    cited_chunk_ids: list[int] = Field(
        default_factory=list,
        description="DB ids of the historical chunks that grounded this component.",
    )
    rationale: str = Field(description="Why this number, in one or two sentences.")


class AgentEstimate(BaseModel):
    """The agent's final structured estimate (light — no mandatory citations)."""

    components: list[AgentComponent]
    total_hours: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Confidence


class AgentRunResult(BaseModel):
    """Everything a single agent run produces: the estimate plus its trace."""

    estimate: AgentEstimate | None = Field(
        default=None,
        description="None when the loop stopped before producing a parseable estimate.",
    )
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Number of Responses API round-trips.")
    stopped_reason: Literal["completed", "max_iterations", "no_final_estimate"] = (
        "completed"
    )
