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

# Default window size; a turn is one user message plus its assistant reply.
MAX_TURNS = 6

SUMMARY_PREFIX = "[Earlier conversation summary — the recent turns below are the live thread]\n"


class SessionNotFoundError(KeyError):
    """Raised when a ``session_id`` is not in the store."""


class Message(BaseModel):
    role: Role
    content: str


class ConversationHistory(BaseModel):
    """Sliding window of recent turns, augmented with a cumulative summary and anchors.

    - ``messages``: the recent window (last ``max_turns`` turns once compressed).
    - ``anchors``: turns flagged as durable commitments (NDA, frozen scope, ...);
      kept verbatim and never evicted.
    - ``summary``: running summary of older non-anchor turns already evicted.

    ``add`` only appends: trimming, anchor promotion and summarisation are the job of
    ``app.sessions.compression.CompressionPolicy``. The system prompt is kept apart,
    so it never takes a slot and can never be evicted.
    """

    max_turns: int = Field(default=MAX_TURNS, ge=1)
    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    anchors: list[Message] = Field(default_factory=list)
    summary: str | None = None

    def add(self, role: Role, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def to_messages_list(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        """LLM-ready messages: ``[system?] + [summary?] + anchors + recent window``.

        Pass ``system_prompt`` rendered from the current ``ProjectMetadata``; it replaces
        the previous one and is remembered, so the history always has one.
        """
        if system_prompt is not None:
            self.system_prompt = system_prompt
        out: list[dict[str, str]] = []
        if self.system_prompt is not None:
            out.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            out.append({"role": "user", "content": SUMMARY_PREFIX + self.summary})
        out.extend({"role": m.role, "content": m.content} for m in self.anchors)
        out.extend({"role": m.role, "content": m.content} for m in self.messages)
        return out


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
    turn_count: int = Field(default=0, exclude=True)
    # Cached from the last successful turn so GET /sessions/{id} need not re-run the resolver.
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None


class SessionStore:
    """``session_id`` -> ``Session`` dict held in process memory (see module docstring)."""

    def __init__(self, *, max_turns: int = MAX_TURNS) -> None:
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
