"""Safe, client-facing reason for a failed LLM call.

Provider exceptions can embed API key fragments or account details, so only a fixed
code is derived from them. The single free-text message allowed is the wording of our
own validators, with the model output (``input``) stripped out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import openai
from instructor.core import IncompleteOutputException, InstructorRetryException
from litellm.exceptions import ContextWindowExceededError, ServiceUnavailableError
from pydantic import ValidationError

from app.services.llm_wrapper import LLMTruncatedResponseError

MAX_REASON_MESSAGE_CHARS = 300
MAX_VALIDATION_ISSUES = 3


@dataclass(frozen=True)
class LLMFailureReason:
    code: str
    message: str | None = None

    def as_dict(self) -> dict[str, str]:
        if self.message is None:
            return {"code": self.code}
        return {"code": self.code, "message": self.message}


PROVIDER_ERROR = LLMFailureReason("provider_error")
UNEXPECTED_ERROR = LLMFailureReason("unexpected_error")


def classify_llm_error(exc: BaseException) -> LLMFailureReason:
    root = _root_cause(exc)
    if isinstance(root, ValidationError):
        return LLMFailureReason("validation_failed", _validation_message(root))
    if isinstance(
        root, (json.JSONDecodeError, IncompleteOutputException, LLMTruncatedResponseError)
    ):
        return LLMFailureReason("incomplete_response", "The model returned an incomplete or invalid response.")
    if isinstance(root, ContextWindowExceededError):
        return LLMFailureReason("context_too_long")
    if isinstance(root, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return LLMFailureReason("authentication_failed")
    if isinstance(root, openai.RateLimitError):
        return LLMFailureReason("rate_limited")
    # APITimeoutError subclasses APIConnectionError, so it must be checked first.
    if isinstance(root, openai.APITimeoutError):
        return LLMFailureReason("timeout")
    if isinstance(
        root, (openai.APIConnectionError, openai.InternalServerError, ServiceUnavailableError)
    ):
        return LLMFailureReason("provider_unavailable")
    if isinstance(root, openai.APIError):
        return PROVIDER_ERROR
    return UNEXPECTED_ERROR


def _root_cause(exc: BaseException) -> BaseException:
    """Instructor wraps whatever stopped the retries; unwrap it to see the real failure."""
    if isinstance(exc, InstructorRetryException):
        attempts = getattr(exc, "failed_attempts", None)
        if attempts:
            return attempts[-1].exception
        if exc.__cause__ is not None:
            return exc.__cause__
    return exc


def _validation_message(error: ValidationError) -> str:
    issues = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False)[
        :MAX_VALIDATION_ISSUES
    ]:
        message = issue["msg"].removeprefix("Value error, ")
        location = ".".join(str(part) for part in issue["loc"])
        issues.append(f"{location}: {message}" if location else message)
    return "; ".join(issues)[:MAX_REASON_MESSAGE_CHARS]
