"""Unit tests for the error-sanitisation contract in app.services.llm_service.

Provider exceptions must never reach the caller verbatim (they can embed API key
fragments, account URLs, etc.), while genuine bugs in our own code must NOT be
relabelled as provider failures.
"""

import pytest
from litellm.exceptions import APIError

from app.services import llm_service
from app.services.llm_service import (
    LLM_PROVIDER_ERROR_MESSAGE,
    GenerationOptions,
    LLMServiceError,
    generate_estimation,
)
from app.services.llm_wrapper import LLMTruncatedResponseError

SECRET_FRAGMENT = "sk-proj-super-secret-fragment"


def test_provider_api_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs) -> dict:
        raise APIError(
            status_code=401, message=SECRET_FRAGMENT, llm_provider="openai", model="gpt-4o-mini"
        )

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    with pytest.raises(LLMServiceError) as exc_info:
        generate_estimation("We need a small CRM with auth. MVP six weeks.")

    assert str(exc_info.value) == LLM_PROVIDER_ERROR_MESSAGE
    assert SECRET_FRAGMENT not in str(exc_info.value)


def test_truncated_response_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs) -> dict:
        raise LLMTruncatedResponseError(
            "Provider stopped before completing the response (finish_reason='length')"
        )

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    with pytest.raises(LLMServiceError) as exc_info:
        generate_estimation("We need a small CRM with auth. MVP six weeks.")

    assert str(exc_info.value) == LLM_PROVIDER_ERROR_MESSAGE


def test_own_bug_is_not_relabelled_as_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs) -> dict:
        raise KeyError("usage")

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    # A bug in our own code must surface as itself, not as a disguised LLMServiceError.
    with pytest.raises(KeyError):
        generate_estimation("We need a small CRM with auth. MVP six weeks.")


def test_extraction_provider_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs) -> dict:
        raise APIError(
            status_code=500, message=SECRET_FRAGMENT, llm_provider="anthropic", model="claude"
        )

    monkeypatch.setattr(llm_service, "_invoke_llm", boom)

    with pytest.raises(LLMServiceError) as exc_info:
        generate_estimation(
            "We need a small CRM with auth. MVP six weeks.",
            GenerationOptions(preprocessing="two_phase"),
        )

    assert str(exc_info.value) == LLM_PROVIDER_ERROR_MESSAGE
    assert SECRET_FRAGMENT not in str(exc_info.value)
