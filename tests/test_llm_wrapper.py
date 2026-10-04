import json
from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
import litellm
import pytest

from app.services.cache import EstimationCache
from app.schemas.estimation import EstimationResult
from app.services.llm_wrapper import LLMTruncatedResponseError, LLMWrapper, _estimate_cost


def _fake_completion(model: str, content: str = "the answer", input_tokens: int = 100, output_tokens: int = 50):
    """Build a SimpleNamespace shaped like a litellm.ModelResponse."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


@pytest.fixture
def wrapper() -> LLMWrapper:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LLMWrapper(
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=2,
        cache=cache,
    )


def _structured_result() -> EstimationResult:
    return EstimationResult(
        summary="Build a booking system with payments.",
        confidence_pct=80,
        phases=[{
            "name": "Implementation", "duration_weeks": 4, "cost_eur": 10000,
            "summary": "Build booking and payment integrations.",
        }],
        total_duration_weeks=4,
        total_cost_eur=10000,
    )


def test_complete_structured_validates_and_replays_cached_result(wrapper: LLMWrapper) -> None:
    from unittest.mock import Mock

    result = _structured_result()
    client = Mock()
    client.chat.completions.create.return_value = result
    with patch("app.services.llm_wrapper.instructor.from_litellm", return_value=client):
        first, meta = wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
        )
        second, cached_meta = wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
        )
    assert first == second == result
    assert meta["cache_hit"] is False
    assert cached_meta["cache_hit"] is True
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["response_model"] is EstimationResult
    assert client.chat.completions.create.call_args.kwargs["max_retries"] == 6


@pytest.mark.parametrize(
    ("model", "key", "mode"),
    [("gpt-4o-mini", "fake-openai", "JSON_SCHEMA"),
     ("claude-haiku-4-5-20251001", "fake-anthropic", "TOOLS")],
)
def test_complete_structured_selects_provider_mode_and_key(
    wrapper: LLMWrapper, model: str, key: str, mode: str,
) -> None:
    import instructor
    from unittest.mock import Mock

    client = Mock()
    client.chat.completions.create.return_value = _structured_result()
    with patch("app.services.llm_wrapper.instructor.from_litellm", return_value=client) as factory:
        wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
            model_override=model,
        )
    assert factory.call_args.kwargs["mode"] == getattr(instructor.Mode, mode)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == model
    assert kwargs["api_key"] == key
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "usr"},
    ]


def test_instructor_retries_invalid_costs_then_caches_valid_result(wrapper: LLMWrapper) -> None:
    valid = _structured_result().model_dump(mode="json")
    invalid = {**valid, "total_cost_eur": 10001}
    responses = [
        litellm.ModelResponse(
            model="gpt-4o-mini",
            choices=[{
                "message": {"role": "assistant", "content": json.dumps(payload)},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        for payload in [invalid, valid]
    ]
    with patch("app.services.llm_wrapper.litellm.completion", side_effect=responses) as provider:
        result, meta = wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
        )
        cached, cached_meta = wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
        )
    assert provider.call_count == 2
    assert result == cached == _structured_result()
    assert meta["cache_hit"] is False
    assert cached_meta["cache_hit"] is True
    assert provider.call_args.kwargs["response_format"]["type"] == "json_schema"


def test_instructor_anthropic_tool_response_is_validated(wrapper: LLMWrapper) -> None:
    response = litellm.ModelResponse(
        model="claude-haiku-4-5-20251001",
        choices=[{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_test", "type": "function",
                    "function": {
                        "name": "EstimationResult",
                        "arguments": _structured_result().model_dump_json(),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    with patch("app.services.llm_wrapper.litellm.completion", return_value=response) as provider:
        result, meta = wrapper.complete_structured(
            system_prompt="sys", user_message="usr", response_model=EstimationResult,
            model_override="claude-haiku-4-5-20251001",
        )
    assert result == _structured_result()
    assert meta["provider"] == "anthropic"
    assert provider.call_args.kwargs["tools"][0]["function"]["name"] == "EstimationResult"
    assert provider.call_args.kwargs["tool_choice"]["function"]["name"] == "EstimationResult"


def test_complete_structured_does_not_cache_failed_generation(wrapper: LLMWrapper) -> None:
    from unittest.mock import Mock

    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("validation failed")
    with patch("app.services.llm_wrapper.instructor.from_litellm", return_value=client):
        for attempt in range(2):
            with pytest.raises(RuntimeError, match="validation failed"):
                wrapper.complete_structured(
                    system_prompt="sys", user_message="usr", response_model=EstimationResult,
                )
    assert client.chat.completions.create.call_count == 2
    assert wrapper.cache.redis.dbsize() == 0


def test_estimate_cost_uses_pricing_table() -> None:
    cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    # 1M input * 0.15 + 1M output * 0.60 = 0.75 USD
    assert cost == pytest.approx(0.75)


def _router_keys_by_model(wrapper: LLMWrapper) -> dict[str, str | None]:
    return {
        deployment["litellm_params"]["model"]: deployment["litellm_params"].get("api_key")
        for deployment in wrapper.router.model_list
    }


def test_router_wires_api_key_by_the_configured_models_actual_provider() -> None:
    """PRIMARY_MODEL/FALLBACK_MODEL can be set to either provider in either slot;
    the correct key must follow the model string, not the primary/fallback position.
    """
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    wrapper = LLMWrapper(
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
        primary_model="claude-haiku-4-5-20251001",  # swapped: primary is Anthropic here
        fallback_model="gpt-4o-mini",  # and fallback is OpenAI
        timeout=30,
        num_retries=2,
        cache=cache,
    )
    keys_by_model = _router_keys_by_model(wrapper)
    assert keys_by_model["claude-haiku-4-5-20251001"] == "anthropic-key"
    assert keys_by_model["gpt-4o-mini"] == "openai-key"


def test_router_leaves_api_key_none_when_that_providers_key_is_missing() -> None:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    wrapper = LLMWrapper(
        openai_api_key=None,
        anthropic_api_key="anthropic-key",
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=2,
        cache=cache,
    )
    keys_by_model = _router_keys_by_model(wrapper)
    assert keys_by_model["gpt-4o-mini"] is None
    assert keys_by_model["claude-haiku-4-5-20251001"] == "anthropic-key"


def test_complete_sends_examples_in_system_role_and_transcription_in_user_role(
    wrapper: LLMWrapper,
) -> None:
    """Pins the data/instructions separation the CAG prompt relies on: whatever
    the caller passes as system_prompt (the CAG examples) must reach the SDK as
    the system message, and the raw transcription must reach it, unmodified, as
    the user message \u2014 in that order.
    """
    fake = _fake_completion(model="gpt-4o-mini", content="ok")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        wrapper.complete(
            system_prompt="ROLE + CAG EXAMPLES BLOCK",
            user_message="raw meeting transcription, verbatim",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    messages = mocked.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "ROLE + CAG EXAMPLES BLOCK"
    assert messages[1]["content"] == "raw meeting transcription, verbatim"


def test_complete_returns_normalised_dict_and_caches(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o-mini", content="hello world")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked.call_count == 1
    assert result["estimation"] == "hello world"
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["cache_hit"] is False
    assert result["cost_usd"] > 0

    # Second call with the same inputs should hit the cache without invoking the router.
    with patch.object(wrapper.router, "completion") as mocked_again:
        cached = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked_again.call_count == 0
    assert cached["cache_hit"] is True
    assert cached["estimation"] == "hello world"


def test_complete_raises_and_skips_cache_when_truncated(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o-mini", content="partial answer")
    fake.choices[0].finish_reason = "length"

    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        with pytest.raises(LLMTruncatedResponseError):
            wrapper.complete(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
                thinking_budget=None,
            )
    assert mocked.call_count == 1

    # The truncated response must never be cached, so a retry hits the provider again.
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked_again:
        with pytest.raises(LLMTruncatedResponseError):
            wrapper.complete(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
                thinking_budget=None,
            )
    assert mocked_again.call_count == 1


def test_complete_with_model_override_bypasses_router(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="gpt-4o", content="overridden")
    with patch("app.services.llm_wrapper.litellm.completion", return_value=fake) as direct, \
        patch.object(wrapper.router, "completion") as router_call:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="gpt-4o",
            max_tokens=4000,
            thinking_budget=None,
        )
    assert direct.call_count == 1
    assert router_call.call_count == 0
    assert direct.call_args.kwargs["model"] == "gpt-4o"
    assert result["model"] == "gpt-4o"


def test_thinking_budget_passed_for_anthropic_fallback(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=2048,
        )
    # primary is OpenAI (gpt-4o-mini), so thinking budget is *ignored* in kwargs.
    assert "thinking" not in mocked.call_args.kwargs


def test_thinking_budget_pads_max_tokens_when_anthropic_override(wrapper: LLMWrapper) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch("app.services.llm_wrapper.litellm.completion", return_value=fake) as direct:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="claude-haiku-4-5-20251001",
            max_tokens=1000,
            thinking_budget=4096,
        )
    kwargs = direct.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kwargs["max_tokens"] == 4096 + 1024


def test_complete_stream_yields_chunks_and_caches(wrapper: LLMWrapper) -> None:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "), finish_reason=None)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="world"), finish_reason=None)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")]
        ),
    ]
    with patch.object(wrapper.router, "completion", return_value=iter(chunks)):
        emitted = list(
            wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
            )
        )
    assert "".join(emitted) == "Hello world"

    # Now the same request hits the cache and replays the full text as one chunk.
    with patch.object(wrapper.router, "completion") as router_call:
        replayed = list(
            wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
            )
        )
    assert router_call.call_count == 0
    assert "".join(replayed) == "Hello world"


def test_complete_stream_raises_and_skips_cache_when_truncated(wrapper: LLMWrapper) -> None:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "), finish_reason=None)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")]
        ),
    ]
    with patch.object(wrapper.router, "completion", return_value=iter(chunks)):
        emitted: list[str] = []
        with pytest.raises(LLMTruncatedResponseError):
            for delta in wrapper.complete_stream(
                system_prompt="sys",
                user_message="usr",
                model_override=None,
                max_tokens=4000,
            ):
                emitted.append(delta)
    # Tokens already yielded before the cut cannot be un-sent, but nothing gets cached.
    assert emitted == ["Hello "]

    with patch.object(wrapper.router, "completion", return_value=iter(chunks)) as mocked_again:
        with pytest.raises(LLMTruncatedResponseError):
            list(
                wrapper.complete_stream(
                    system_prompt="sys",
                    user_message="usr",
                    model_override=None,
                    max_tokens=4000,
                )
            )
    assert mocked_again.call_count == 1
