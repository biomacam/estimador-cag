"""Unit tests for ``CompressionPolicy``, ``CumulativeSummarizer`` and ``apply_compression``."""

from app.sessions import ConversationHistory
from app.sessions.compression import CompressionPolicy, CumulativeSummarizer, apply_compression
from app.sessions.compression.anchors import AnchorDetector
from app.sessions.compression.summarizer import _SummaryEnvelope


class _SummaryLlm:
    def __init__(self, text: str = "Running summary.", error: Exception | None = None) -> None:
        self.text, self.error, self.calls = text, error, []

    def complete_structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _SummaryEnvelope(summary=self.text), {"model": "cheap", "latency_ms": 1}


def _policy(llm: _SummaryLlm) -> CompressionPolicy:
    return CompressionPolicy(
        anchor_detector=AnchorDetector(),
        summarizer=CumulativeSummarizer(llm_wrapper=llm, model="cheap"),
    )


def _history(max_turns: int, *users: str) -> ConversationHistory:
    history = ConversationHistory(max_turns=max_turns)
    for n, user in enumerate(users, start=1):
        history.add("user", user)
        history.add("assistant", f"a{n}")
    return history


def test_no_op_when_window_is_under_the_cap() -> None:
    llm = _SummaryLlm()
    history = _history(3, "u1", "u2")

    _policy(llm).apply(history)

    assert llm.calls == []
    assert history.summary is None
    assert history.anchors == []
    assert len(history.messages) == 4


def test_non_anchor_overflow_is_folded_into_the_summary() -> None:
    llm = _SummaryLlm("t1 absorbed")
    history = _history(2, "t1 plain", "t2 plain", "t3 plain")

    _policy(llm).apply(history)

    assert history.summary == "t1 absorbed"
    assert history.anchors == []
    assert [m.content for m in history.messages] == ["t2 plain", "a2", "t3 plain", "a3"]
    summarizer_input = llm.calls[0]["messages"][1]["content"]
    assert "[user] t1 plain" in summarizer_input
    assert "[assistant] a1" in summarizer_input
    assert llm.calls[0]["model_override"] == "cheap"


def test_anchor_overflow_is_promoted_verbatim_and_skips_the_summarizer() -> None:
    llm = _SummaryLlm()
    history = _history(2, "We signed the NDA last week.", "t2 plain", "t3 plain")

    _policy(llm).apply(history)

    assert [m.content for m in history.anchors] == ["We signed the NDA last week.", "a1"]
    assert history.summary is None
    assert llm.calls == []
    assert [m.content for m in history.messages] == ["t2 plain", "a2", "t3 plain", "a3"]


def test_previous_summary_is_passed_to_the_next_pass() -> None:
    llm = _SummaryLlm("second pass")
    history = _history(1, "t1 plain", "t2 plain")
    history.summary = "first pass"

    _policy(llm).apply(history)

    assert history.summary == "second pass"
    assert "first pass" in llm.calls[0]["messages"][1]["content"]


def test_to_messages_list_composes_system_summary_anchors_and_recent_window() -> None:
    history = _history(1, "We signed the NDA.", "t2 plain", "t3 plain")
    apply_compression(history, llm_wrapper=_SummaryLlm("summary text"), compression_model="cheap")

    messages = history.to_messages_list("system prompt")

    assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "user", "assistant"]
    assert messages[1]["content"].endswith("summary text")
    assert messages[2]["content"] == "We signed the NDA."
    assert messages[-2]["content"] == "t3 plain"


def test_repeated_apply_without_new_messages_is_idempotent() -> None:
    llm = _SummaryLlm("once")
    history = _history(2, *(f"t{n} plain" for n in range(5)))
    policy = _policy(llm)

    policy.apply(history)
    snapshot = history.model_copy(deep=True)
    policy.apply(history)

    assert history == snapshot
    assert len(llm.calls) == 1


def test_summarizer_failure_keeps_the_previous_summary_but_still_trims() -> None:
    history = _history(2, "t1", "t2", "t3")
    history.summary = "Existing summary stays."

    _policy(_SummaryLlm(error=RuntimeError("LLM down"))).apply(history)

    assert history.summary == "Existing summary stays."
    assert len(history.messages) == 4
