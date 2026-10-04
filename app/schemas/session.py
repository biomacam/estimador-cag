from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID v4 to send with every later request of this conversation")
