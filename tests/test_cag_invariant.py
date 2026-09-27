"""The one invariant this whole exercise exists to prove: the canonical CAG
examples actually reach the LLM's system prompt, not just the format helpers
in isolation. A test that only checks HTTP status codes would pass even if
``build_system_prompt`` never imported ``CANONICAL_EXAMPLES`` at all.
"""

from app.context.examples import CANONICAL_EXAMPLES
from app.services.llm_service import build_system_prompt


def test_examples_reach_the_system_prompt() -> None:
    prompt = build_system_prompt()

    for example in CANONICAL_EXAMPLES[:3]:  # build_system_prompt()'s default num_examples
        assert example.estimation_markdown in prompt


def test_num_examples_controls_how_many_reach_the_prompt() -> None:
    prompt = build_system_prompt(num_examples=1)

    assert CANONICAL_EXAMPLES[0].estimation_markdown in prompt
    assert CANONICAL_EXAMPLES[1].estimation_markdown not in prompt


def test_use_examples_false_leaves_no_example_content_in_prompt() -> None:
    prompt = build_system_prompt(use_examples=False)

    for example in CANONICAL_EXAMPLES:
        assert example.estimation_markdown not in prompt
