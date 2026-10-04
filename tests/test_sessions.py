import pytest
from pydantic import ValidationError

from app.sessions import (
    ConversationHistory,
    ProjectMetadata,
    SessionNotFoundError,
    SessionStore,
)


def _add_turns(history: ConversationHistory, count: int) -> None:
    for n in range(1, count + 1):
        history.add("user", f"u{n}")
        history.add("assistant", f"a{n}")


def test_window_drops_the_oldest_turns_beyond_max_turns() -> None:
    history = ConversationHistory(max_turns=2)

    _add_turns(history, 3)

    assert [m.content for m in history.messages] == ["u2", "a2", "u3", "a3"]


def test_window_never_exceeds_max_turns_over_many_turns() -> None:
    history = ConversationHistory(max_turns=3)

    _add_turns(history, 10)

    assert len(history.messages) == 6
    assert history.messages[0].content == "u8"


def test_pending_user_message_counts_as_a_turn_and_keeps_roles_aligned() -> None:
    history = ConversationHistory(max_turns=2)
    _add_turns(history, 2)

    history.add("user", "u3")

    assert [m.content for m in history.messages] == ["u2", "a2", "u3"]
    assert history.messages[0].role == "user"


def test_system_prompt_is_always_first_and_survives_trimming() -> None:
    history = ConversationHistory(max_turns=1, system_prompt="You are an estimator.")

    _add_turns(history, 5)

    messages = history.to_messages()
    assert messages[0] == {"role": "system", "content": "You are an estimator."}
    assert [m["content"] for m in messages[1:]] == ["u5", "a5"]


def test_to_messages_has_no_system_entry_when_prompt_is_unset() -> None:
    history = ConversationHistory()
    history.add("user", "hello")

    assert history.to_messages() == [{"role": "user", "content": "hello"}]


def test_max_turns_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ConversationHistory(max_turns=0)


def test_project_metadata_starts_empty_and_lists_are_not_shared() -> None:
    first, second = ProjectMetadata(), ProjectMetadata()

    first.mentioned_technologies.append("FastAPI")

    assert first.project_name is None
    assert second.mentioned_technologies == []


def test_project_metadata_rejects_non_positive_team_size() -> None:
    with pytest.raises(ValidationError):
        ProjectMetadata(assumed_team_size=0)


def test_store_creates_independent_sessions_with_configured_window() -> None:
    store = SessionStore(max_turns=4)

    first, second = store.create(), store.create()
    first.metadata.project_name = "CRM"

    assert first.session_id != second.session_id
    assert second.metadata.project_name is None
    assert first.history.max_turns == 4
    assert len(store) == 2


def test_store_get_returns_the_same_session_object() -> None:
    store = SessionStore()
    session = store.create()

    assert store.get(session.session_id) is session


def test_store_get_raises_for_unknown_id() -> None:
    with pytest.raises(SessionNotFoundError):
        SessionStore().get("missing")


def test_project_metadata_is_empty_only_without_any_fact() -> None:
    assert ProjectMetadata().is_empty()
    assert not ProjectMetadata(mentioned_technologies=["FastAPI"]).is_empty()


def test_merge_overwrites_non_null_scalars_and_keeps_the_rest() -> None:
    base = ProjectMetadata(project_name="CRM", assumed_team_size=3, agreed_scope="Phase 1")

    merged = base.merge_with(ProjectMetadata(project_name="CRM v2"))

    assert merged.project_name == "CRM v2"
    assert merged.assumed_team_size == 3
    assert merged.agreed_scope == "Phase 1"
    assert base.project_name == "CRM"


def test_merge_unions_technologies_case_insensitively_keeping_order() -> None:
    base = ProjectMetadata(mentioned_technologies=["React", "Postgres"])

    merged = base.merge_with(ProjectMetadata(mentioned_technologies=["postgres", "Redis"]))

    assert merged.mentioned_technologies == ["React", "Postgres", "Redis"]


def test_project_metadata_bounds_what_an_llm_can_write() -> None:
    with pytest.raises(ValidationError):
        ProjectMetadata(assumed_team_size=51)
    with pytest.raises(ValidationError):
        ProjectMetadata(project_name="x" * 121)
