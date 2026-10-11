"""Unit tests for the runtime audience tier resolver."""

from app.sessions import ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier


def test_default_tier_for_neutral_input() -> None:
    tier, rule = resolve_tier(
        transcript="A CRM for the sales team with contacts and pipelines.",
        metadata=ProjectMetadata(),
    )
    assert (tier, rule) == (Tier.DEFAULT, "default")


def test_explicit_override_beats_every_rule() -> None:
    tier, rule = resolve_tier(
        transcript="We signed the NDA and run HIPAA workloads.",
        metadata=ProjectMetadata(),
        override=Tier.PM,
    )
    assert (tier, rule) == (Tier.PM, "explicit_override")


def test_nda_in_transcript_promotes_to_executive() -> None:
    tier, rule = resolve_tier(
        transcript="Heads-up, we signed an NDA before this call.", metadata=ProjectMetadata()
    )
    assert (tier, rule) == (Tier.EXECUTIVE, "nda_detected")


def test_nda_in_agreed_scope_promotes_to_executive() -> None:
    tier, rule = resolve_tier(
        transcript="Just a follow up.",
        metadata=ProjectMetadata(agreed_scope="Under embargo until Q4."),
    )
    assert (tier, rule) == (Tier.EXECUTIVE, "nda_detected")


def test_regulatory_keyword_promotes_to_executive() -> None:
    tier, rule = resolve_tier(
        transcript="The platform must be HIPAA-compliant from day one.", metadata=ProjectMetadata()
    )
    assert (tier, rule) == (Tier.EXECUTIVE, "regulatory_context")


def test_regulatory_term_in_metadata_technologies_promotes() -> None:
    tier, rule = resolve_tier(
        transcript="Migrate to a new payments stack.",
        metadata=ProjectMetadata(mentioned_technologies=["Stripe", "PCI-DSS"]),
    )
    assert (tier, rule) == (Tier.EXECUTIVE, "regulatory_context")


def test_two_or_more_dev_keywords_promote_to_developer() -> None:
    tier, rule = resolve_tier(
        transcript="We run microservices on Kubernetes with a Terraform pipeline.",
        metadata=ProjectMetadata(),
    )
    assert (tier, rule) == (Tier.DEVELOPER, "technical_audience")


def test_single_dev_keyword_is_not_enough_for_developer() -> None:
    tier, _ = resolve_tier(
        transcript="We deploy with docker today, Docker everywhere.", metadata=ProjectMetadata()
    )
    assert tier is Tier.DEFAULT


def test_small_team_metadata_promotes_to_pm() -> None:
    tier, rule = resolve_tier(
        transcript="Just two of us building this MVP.",
        metadata=ProjectMetadata(assumed_team_size=2),
    )
    assert (tier, rule) == (Tier.PM, "low_budget_pm")


def test_nda_beats_small_team() -> None:
    tier, rule = resolve_tier(
        transcript="Just two of us — and we signed an NDA last week.",
        metadata=ProjectMetadata(assumed_team_size=2),
    )
    assert (tier, rule) == (Tier.EXECUTIVE, "nda_detected")
