from pydantic import BaseModel, Field

from app.sessions import ProjectMetadata


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID v4 to send with every later request of this conversation")


class SessionInfoResponse(BaseModel):
    """Read-only view of a session: the memory (metadata) apart from the history size."""

    session_id: str
    message_count: int = Field(description="Messages currently inside the sliding window")
    max_turns: int
    metadata: ProjectMetadata
