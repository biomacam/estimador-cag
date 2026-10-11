"""Unit tests for the anchor detector (heuristic and LLM modes)."""

import pytest

from app.sessions import Message
from app.sessions.compression.anchors import AnchorDetector, _AnchorClassification


def _user(content: str) -> Message:
    return Message(role="user", content=content)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("We signed the NDA last week.", "nda"),
        ("The client countersigned the SOW yesterday.", "signed_contract"),
        ("Scope is frozen for phase 1.", "scope_frozen"),
        ("Budget is capped at 50k EUR.", "budget_locked"),
        ("Patient data falls under HIPAA.", "compliance"),
        ("There is a hard deadline in March.", "deadline_hard"),
        ("We are contractually bound to deliver SSO.", "contractual"),
        ("The client agreed to drop the mobile app.", "explicit_commitment"),
    ],
)
def test_heuristic_flags_durable_commitments(text: str, rule: str) -> None:
    match = AnchorDetector().detect(_user(text))

    assert match.is_anchor
    assert rule in match.matched_rules


def test_heuristic_ignores_regular_scope_talk() -> None:
    match = AnchorDetector().detect(_user("Let's add a reporting dashboard with React."))

    assert not match.is_anchor
    assert match.matched_rules == []


class _Classifier:
    def __init__(self, *, is_anchor: bool | None = None, error: Exception | None = None) -> None:
        self.is_anchor, self.error, self.calls = is_anchor, error, []

    def complete_structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _AnchorClassification(is_anchor=self.is_anchor, reason="locked budget"), {}


def test_llm_mode_uses_the_classifier_and_frames_the_turn_as_data() -> None:
    llm = _Classifier(is_anchor=True)

    match = AnchorDetector(mode="llm", llm_wrapper=llm, llm_model="cheap").detect(
        _user("We can't spend more than what finance already approved.")
    )

    assert match.is_anchor
    assert match.matched_rules[0].startswith("llm:")
    call = llm.calls[0]
    assert call["model_override"] == "cheap"
    assert "<user-data-" in call["messages"][1]["content"]


def test_llm_mode_falls_back_to_heuristic_on_error() -> None:
    detector = AnchorDetector(mode="llm", llm_wrapper=_Classifier(error=RuntimeError("down")))

    assert detector.detect(_user("We signed the NDA.")).is_anchor
    assert not detector.detect(_user("Plain scope talk.")).is_anchor


def test_llm_mode_without_wrapper_behaves_like_heuristic() -> None:
    assert AnchorDetector(mode="llm").detect(_user("GDPR applies here.")).is_anchor
