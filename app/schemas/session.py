from pydantic import BaseModel, Field

from app.sessions import Message, ProjectMetadata


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID v4 to send with every later request of this conversation")


class SessionInfoResponse(BaseModel):
    """Read-only view of a session: the memory (metadata) apart from the history size."""

    session_id: str
    message_count: int = Field(description="Messages currently inside the sliding window")
    max_turns: int
    anchors_count: int = Field(description="Anchor messages kept verbatim outside the window")
    summary_chars: int = Field(description="Length of the cumulative summary of evicted turns")
    last_resolved_tier: str | None = Field(description="Audience tier used in the last successful turn")
    last_tier_rule: str | None = Field(description="Rule that decided that tier (or explicit_override)")
    metadata: ProjectMetadata
    summary: str | None = Field(default=None, description="Cumulative summary text of evicted turns")
    anchors: list[Message] = Field(default_factory=list, description="Anchor messages kept verbatim")
