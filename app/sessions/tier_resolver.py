"""Derive the audience tier at runtime from the conversation context.

Tiers shape the prompt's framing (executive bullets vs PM milestones vs
developer-grade detail vs default). Two ways to set them:

- **Explicit override** by the caller (form field ``tier=executive``). Always wins.
- **Implicit derivation** from the latest transcript + accumulated metadata.

The derivation is a precedence-ordered chain of pure-function rules: the first
match decides the tier; if none matches, the tier is ``DEFAULT``. The resolver
returns both the tier and the rule name that fired, so callers can show
"executive (nda_detected)" instead of just "executive".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import structlog

from app.sessions.models import ProjectMetadata

log = structlog.get_logger()


class Tier(str, Enum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata


@dataclass
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


_NDA_PATTERN = re.compile(
    r"\b(nda|non[- ]?disclosure|confidential|under embargo|legal hold)\b", re.IGNORECASE
)
_REGULATORY_PATTERN = re.compile(
    r"\b(hipaa|gdpr|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b", re.IGNORECASE
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservices?|terraform|iac|"
    r"helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


def _has_nda(ctx: ResolutionContext) -> bool:
    return bool(
        _NDA_PATTERN.search(ctx.transcript)
        or (ctx.metadata.agreed_scope and _NDA_PATTERN.search(ctx.metadata.agreed_scope))
    )


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    texts = [ctx.transcript, ctx.metadata.agreed_scope or "", *ctx.metadata.mentioned_technologies]
    return any(_REGULATORY_PATTERN.search(text) for text in texts)


def _is_small_team(ctx: ResolutionContext) -> bool:
    return ctx.metadata.assumed_team_size is not None and ctx.metadata.assumed_team_size <= 2


def _technical_audience(ctx: ResolutionContext) -> bool:
    # A single mention ("we run on docker") is too noisy to promote the tier.
    hits = _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)
    return len({hit.lower() for hit in hits}) >= 2


_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *, transcript: str, metadata: ProjectMetadata, override: Tier | None = None
) -> tuple[Tier, str]:
    """Return ``(tier, rule_name)``: override wins, else first matching rule, else DEFAULT."""
    if override is not None:
        log.info("tier_resolved", tier=override.value, rule="explicit_override")
        return override, "explicit_override"

    ctx = ResolutionContext(transcript=transcript, metadata=metadata)
    for rule in _RULES:
        try:
            if rule.predicate(ctx):
                log.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
                return rule.tier, rule.name
        except Exception as exc:  # noqa: BLE001 — a broken rule must not fail the turn
            log.warning("tier_rule_predicate_failed", rule=rule.name, error_type=type(exc).__name__)

    log.info("tier_resolved", tier=Tier.DEFAULT.value, rule="default")
    return Tier.DEFAULT, "default"
