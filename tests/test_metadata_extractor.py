import pytest

from app.services import metadata_extractor
from app.services.metadata_extractor import update_metadata
from app.sessions import ProjectMetadata
from tests.helpers import unwrap_untrusted_input, valid_estimation_result


def test_update_metadata_merges_the_extracted_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metadata_extractor,
        "_invoke_extractor",
        lambda messages: (ProjectMetadata(mentioned_technologies=["react", "Redis"]), {"latency_ms": 1}),
    )

    merged = update_metadata(
        previous=ProjectMetadata(project_name="Nimbus", mentioned_technologies=["React"]),
        transcript="Add a Redis cache to the portal.",
        result=valid_estimation_result(),
    )

    assert merged.project_name == "Nimbus"
    assert merged.mentioned_technologies == ["React", "Redis"]


def test_update_metadata_returns_the_previous_metadata_on_any_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(messages):
        raise RuntimeError("provider down")

    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", broken)
    previous = ProjectMetadata(project_name="Nimbus")

    assert update_metadata(
        previous=previous, transcript="Anything", result=valid_estimation_result()
    ) is previous


def test_extractor_receives_the_transcript_framed_as_data(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[dict[str, str]]] = []

    def capture(messages):
        seen.append(messages)
        return ProjectMetadata(), {}

    monkeypatch.setattr(metadata_extractor, "_invoke_extractor", capture)

    update_metadata(
        previous=ProjectMetadata(), transcript="We use Django.", result=valid_estimation_result()
    )

    system, user = seen[0]
    assert system["role"] == "system" and user["role"] == "user"
    framed = user["content"].split("<latest_user_transcript>\n", 1)[1].split(
        "\n</latest_user_transcript>", 1
    )[0]
    assert unwrap_untrusted_input(framed) == "We use Django."


def test_extractor_call_uses_the_configured_model_and_returns_project_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings

    calls: list[dict] = []

    class FakeWrapper:
        def complete_structured_chat(self, **kwargs):
            calls.append(kwargs)
            return ProjectMetadata(project_name="Nimbus"), {"latency_ms": 1}

    monkeypatch.setattr(metadata_extractor, "get_llm_wrapper", lambda: FakeWrapper())
    monkeypatch.setattr(
        metadata_extractor, "get_settings", lambda: Settings(_env_file=None, METADATA_EXTRACTOR_MODEL="gpt-4o-mini")
    )

    extracted, _ = metadata_extractor._invoke_extractor([{"role": "user", "content": "hi"}])

    assert extracted.project_name == "Nimbus"
    assert calls[0]["model_override"] == "gpt-4o-mini"
    assert calls[0]["response_model"] is ProjectMetadata
