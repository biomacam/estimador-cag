import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_do_not_require_an_api_key() -> None:
    """Startup must succeed with no provider configured; /health has to stay reachable."""
    settings = Settings(_env_file=None, OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None)
    assert settings.is_llm_configured is False


def test_settings_report_configured_with_either_key() -> None:
    only_openai = Settings(_env_file=None, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY=None)
    only_anthropic = Settings(_env_file=None, OPENAI_API_KEY=None, ANTHROPIC_API_KEY="sk-ant-test")
    assert only_openai.is_llm_configured is True
    assert only_anthropic.is_llm_configured is True


def test_dead_provider_model_fields_were_removed() -> None:
    """LLM_PROVIDER/LLM_MODEL never controlled anything (PRIMARY_MODEL/FALLBACK_MODEL
    do); keeping them around only invited configuring one and expecting an effect.
    """
    assert "LLM_PROVIDER" not in Settings.model_fields
    assert "LLM_MODEL" not in Settings.model_fields


def test_metadata_extractor_defaults_to_the_cheap_model() -> None:
    assert Settings(_env_file=None).METADATA_EXTRACTOR_MODEL == "gpt-4o-mini"


def test_conversation_and_attachment_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.MAX_CONVERSATION_TURNS == 6
    assert settings.MAX_ATTACHMENT_CHARS == 60_000


def test_conversation_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_CONVERSATION_TURNS=0)
