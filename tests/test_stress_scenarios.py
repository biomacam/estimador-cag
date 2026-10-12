import pytest

from evals.stress.scenarios import (
    FULL_LENGTH,
    SCENARIO_LENGTHS,
    all_scenarios,
    get_scenario,
)


@pytest.mark.parametrize("scenario", all_scenarios(), ids=lambda s: s.name)
def test_every_profile_is_a_numbered_20_turn_script(scenario):
    assert [t.turn_index for t in scenario.turns] == list(range(1, FULL_LENGTH + 1))
    # The sessions endpoint rejects transcripts shorter than 20 chars.
    assert all(len(t.transcript) >= 20 for t in scenario.turns)
    for n in SCENARIO_LENGTHS:
        assert scenario.first(n) == scenario.turns[:n]


def test_unsupported_length_is_rejected():
    with pytest.raises(ValueError):
        get_scenario("growing").first(5)


def test_growing_introduces_the_project_name_at_turn_1():
    turn = get_scenario("growing").turns[0]
    assert (turn.fact_to_remember, turn.fact_field) == ("Nimbus", "project_name")
    assert "Nimbus" in turn.transcript


def test_pivot_switches_stack_at_turn_5():
    scenario = get_scenario("pivot")
    assert "React Native" in scenario.turns[0].transcript
    pivot = scenario.turns[4]
    assert pivot.turn_index == 5 and pivot.fact_to_remember == "Flutter"
    assert "Flutter" not in " ".join(t.transcript for t in scenario.turns[:4])


def test_contradiction_states_30k_at_turn_3_and_80k_at_turn_8():
    scenario = get_scenario("contradiction")
    assert "30000" in scenario.turns[2].transcript
    assert "80000" in scenario.turns[7].transcript
    assert [t.turn_index for t in scenario.facts_before(4) if t.fact_to_remember == "30"] == [3]
