"""In-process conversational session state.

Sessions live in a plain dict inside the API process: no database, no Redis.
That volatility is accepted in this phase because the goal is to validate the
conversational flow (sliding window, project metadata) before paying for
persistence. The cost is that a restart or an uvicorn ``--reload`` drops every
session, and with several workers each one would hold its own separate copy.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class SessionNotFoundError(KeyError):
    """Raised when a ``session_id`` is not in the store."""


class Message(BaseModel):
    role: Role
    content: str


class ConversationHistory(BaseModel):
    """Sliding window over the last ``max_turns`` turns of a conversation.

    A turn starts at each user message and includes the assistant reply that
    follows it. The system prompt is kept apart from ``messages``, so trimming
    can never discard it; it is always emitted first by ``to_messages``.
    """

    max_turns: int = Field(default=6, ge=1)
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self._trim()

    def to_messages(self) -> list[dict[str, str]]:
        """Return the LLM-ready messages: system prompt first, then the window."""
        window = [{"role": m.role, "content": m.content} for m in self.messages]
        if self.system_prompt is None:
            return window
        return [{"role": "system", "content": self.system_prompt}, *window]

    def _trim(self) -> None:
        while sum(m.role == "user" for m in self.messages) > self.max_turns:
            del self.messages[0]
            while self.messages and self.messages[0].role != "user":
                del self.messages[0]


class ProjectMetadata(BaseModel):
    """Facts about the project under discussion, kept apart from the history.

    Bounded because an LLM fills it in and its values are injected into later prompts.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=50)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=2000)

    def is_empty(self) -> bool:
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def merge_with(self, update: ProjectMetadata) -> ProjectMetadata:
        """Non-null scalars in ``update`` win; technologies accumulate (case-insensitive union)."""
        technologies = list(self.mentioned_technologies)
        seen = {tech.lower() for tech in technologies}
        for tech in update.mentioned_technologies:
            if tech.lower() not in seen:
                technologies.append(tech)
                seen.add(tech.lower())
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


class Session(BaseModel):
    """One conversation: its history window plus its project metadata."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)


class SessionStore:
    """``session_id`` -> ``Session`` dict held in process memory (see module docstring)."""

    def __init__(self, *, max_turns: int = 6) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def __len__(self) -> int:
        return len(self._sessions)
