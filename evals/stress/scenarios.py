"""Three multi-turn synthetic scenarios with fact-trackers.

Each profile is a fixed list of 20 ``ScenarioTurn`` tuples
``(turn_index, transcript, fact_to_remember, fact_field)``. The runner walks them
in order against a single session, so turn N sees the compressed view of turns
1..N-1. Shorter conversations (N in ``SCENARIO_LENGTHS``) are prefixes of the
20-turn script, so they stay comparable with each other.

``fact_to_remember`` is the literal substring ``MemoryDriftMetric`` looks for in the
session snapshot of later turns; ``fact_field`` scopes where it must appear.

The profiles each force one behaviour of the system:

- ``growing``       — requirements accrete turn by turn: cost curve, and whether the
                      turn-1 ``project_name`` survives the sliding window at turn 20.
- ``pivot``         — turn 5 swaps React Native for Flutter: does the metadata update
                      cleanly or does ``mentioned_technologies`` accumulate both?
- ``contradiction`` — turn 3 says 30k EUR, turn 8 says 80k EUR: which one is kept,
                      promoted to an anchor, or left in the summary?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from evals.stress.metrics import FactField

SCENARIO_LENGTHS = (1, 3, 6, 10, 20)
FULL_LENGTH = max(SCENARIO_LENGTHS)


class ScenarioTurn(NamedTuple):
    """One turn: ``(turn_index, transcript, fact_to_remember)`` plus the metric scope."""

    turn_index: int  # 1-based
    transcript: str
    fact_to_remember: str | None = None
    fact_field: FactField = "any"


@dataclass(frozen=True)
class Scenario:
    name: Literal["growing", "pivot", "contradiction"]
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    turns: list[ScenarioTurn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if [t.turn_index for t in self.turns] != list(range(1, len(self.turns) + 1)):
            raise ValueError(f"scenario {self.name!r}: turn_index must be 1..N without gaps")
        if len(self.turns) < FULL_LENGTH:
            raise ValueError(
                f"scenario {self.name!r} has {len(self.turns)} turns; "
                f"at least {FULL_LENGTH} are needed to push the sliding window past its cap"
            )

    def first(self, n: int) -> list[ScenarioTurn]:
        """The first ``n`` turns; ``n`` must be one of ``SCENARIO_LENGTHS``."""
        if n not in SCENARIO_LENGTHS:
            raise ValueError(f"unsupported length {n}; use one of {SCENARIO_LENGTHS}")
        return self.turns[:n]

    def facts_before(self, turn_index: int) -> list[ScenarioTurn]:
        """Turns that introduced a fact earlier than ``turn_index`` (what that call must remember)."""
        return [t for t in self.turns if t.turn_index < turn_index and t.fact_to_remember]


def _number(rows: list[tuple[str, str | None, FactField]]) -> list[ScenarioTurn]:
    return [
        ScenarioTurn(index, transcript, fact, fact_field)
        for index, (transcript, fact, fact_field) in enumerate(rows, start=1)
    ]


# --- growing ----------------------------------------------------------------

_GROWING_TURNS = _number([
    ("We want a B2B SaaS called Nimbus for procurement teams. Stack: React + "
     "Postgres. Team of 3. Initial scope: vendor catalogue, RFQ workflow, "
     "basic dashboards.", "Nimbus", "project_name"),
    ("Add SSO with Okta. Single sign-on must work for both employees and "
     "external vendor users on the same domain.", "Okta", "technologies"),
    ("Confirmed budget: 60k EUR for the first release. Team stays at 3.", "60", "any"),
    ("We need an audit log of every approval, with WORM storage for 7 years "
     "for compliance with internal policy.", "audit log", "any"),
    ("Add CSV import for legacy vendor master data, ~50k rows expected.", "CSV import", "any"),
    ("Multi-tenant: we'll resell this to two more sister companies. Strict "
     "data isolation per tenant.", "multi-tenant", "any"),
    ("Add notification fan-out via email and Microsoft Teams webhook.", "Teams webhook", "technologies"),
    ("Public read-only API for partners to query catalog availability. "
     "OAuth2 client credentials.", "OAuth2", "any"),
    ("Add a reporting module: aggregated KPIs by buyer and category, "
     "weekly digest exported to PDF.", "weekly digest", "any"),
    ("Internationalisation: ES, EN, PT. Currency must follow the tenant.", "ES, EN, PT", "any"),
    ("Mobile-friendly responsive UI for approvers, but no native app yet.", "responsive", "any"),
    ("Background job system for nightly reconciliation against the ERP "
     "(SAP) using IDoc files.", "SAP", "technologies"),
    ("Two-factor auth for any user with approval rights above 10k EUR.", "two-factor", "any"),
    ("Bulk PO export to XML following an UBL profile the procurement team "
     "shared.", "UBL", "any"),
    ("GDPR: right-to-erasure flow for vendor contacts on request.", "GDPR", "any"),
    ("Add a per-tenant feature flag for an experimental AI suggestion "
     "panel that recommends similar past PRs.", "feature flag", "any"),
    ("Vendor self-service portal: vendors update their own banking and "
     "tax info, with admin approval.", "self-service portal", "any"),
    ("Performance SLA: P95 page load under 2s with 200 concurrent users.", "P95", "any"),
    ("Disaster recovery: RPO 1 hour, RTO 4 hours. Daily cross-region "
     "Postgres snapshots.", "RPO", "any"),
    ("Final ask: integration with DocuSign for contract counter-signature "
     "right after vendor approval.", "DocuSign", "technologies"),
])


# --- pivot ------------------------------------------------------------------

_PIVOT_TURNS = _number([
    ("Build a customer-facing app called Helios for a retail chain. "
     "Stack: React Native targeting iOS and Android. Team of 4.", "Helios", "project_name"),
    ("Loyalty card replacement: members scan QR at checkout, accrue points, "
     "redeem in-app.", "loyalty", "any"),
    ("Backend: a Node.js API on top of the existing Postgres customer DB.", "Node.js", "technologies"),
    ("Push notifications for offers segmented by store region.", "push notifications", "any"),
    ("PIVOT: the client just announced they will standardise on Flutter "
     "across all mobile properties. We switch the stack to Flutter. "
     "React Native is OFF the table.", "Flutter", "technologies"),
    ("Re-confirmed Flutter for both iOS and Android; same scope as before.", "Flutter", "technologies"),
    ("Add Apple Wallet and Google Wallet passes for the loyalty card.", "Apple Wallet", "any"),
    ("Integration with the existing PoS API (REST/JSON) for point accrual.", "PoS API", "any"),
    ("Add bar-code scanning for in-store products to show item details.", "bar-code", "any"),
    ("Geofenced offers: deliver a push when a member is near a store.", "geofenced", "any"),
    ("Add social login (Apple ID, Google).", "Apple ID", "any"),
    ("Internal admin panel (web) for marketers to schedule campaigns. "
     "Vue.js for the admin side.", "Vue.js", "technologies"),
    ("Analytics: events to Segment + downstream to BigQuery.", "Segment", "technologies"),
    ("Add in-app messaging for customer support, integrated with Zendesk.", "Zendesk", "technologies"),
    ("Feature: digital receipts that survive even if the printer fails at "
     "the till.", "digital receipts", "any"),
    ("Localisation: ES and CA only; no English in v1.", "ES and CA", "any"),
    ("Crash reporting and performance traces via Firebase Crashlytics.", "Crashlytics", "technologies"),
    ("App store presence: ASO assets and screenshots for 3 device sizes.", "ASO", "any"),
    ("Compliance: GDPR consent flow on first open.", "GDPR", "any"),
    ("Final ask: add an offline mode that lets members view their card "
     "and recent rewards without connectivity.", "offline mode", "any"),
])


# --- contradiction ----------------------------------------------------------

_CONTRADICTION_TURNS = _number([
    ("Build an internal tool called Atlas for the finance department. "
     "Approvals workflow for AP invoices. Stack: Rails + Postgres.", "Atlas", "project_name"),
    ("Team is 2 developers plus a part-time designer.", "2 developers", "any"),
    ("Budget cap announced: 30000 EUR for the whole engagement.", "30", "any"),
    ("Add an audit log of who approved what, exportable to CSV.", "audit log", "any"),
    ("Integration with the existing SSO (Azure AD) — every approver must "
     "authenticate via SSO.", "Azure AD", "technologies"),
    ("Add a dashboard with pending approvals grouped by approver.", "dashboard", "any"),
    ("Bulk approve up to 50 invoices in one action, with a confirmation "
     "step.", "bulk approve", "any"),
    ("CONTRADICTION: leadership revised the budget upward. The new budget "
     "is 80000 EUR; the earlier 30000 figure is obsolete. Plan accordingly.", "80", "any"),
    ("Confirmed new budget: 80000 EUR. The earlier 30k was a placeholder.", "80", "any"),
    ("Add three-way match against POs and goods receipts.", "three-way match", "any"),
    ("Integration with the ERP (Oracle EBS) over its REST gateway.", "Oracle EBS", "technologies"),
    ("Add per-cost-centre routing of invoices to the correct approver.", "cost-centre", "any"),
    ("Email digest every morning summarising pending approvals per approver.", "email digest", "any"),
    ("Add OCR-assisted data entry for paper invoices using AWS Textract.", "Textract", "technologies"),
    ("Support multi-currency invoices with daily ECB rates for normalisation.", "multi-currency", "any"),
    ("Reports module: aging buckets and total exposure per supplier.", "aging buckets", "any"),
    ("Add comments thread on each invoice for approver back-and-forth.", "comments thread", "any"),
    ("Performance: handle 5000 invoices per day at peak month-end.", "5000 invoices", "any"),
    ("Add 4-eyes-principle for invoices above 25000 EUR.", "4-eyes", "any"),
    ("Final ask: nightly export of approved invoices to the data warehouse "
     "(Snowflake).", "Snowflake", "technologies"),
])


_SCENARIOS: dict[str, Scenario] = {
    "growing": Scenario("growing", ProjectType.WEB_SAAS, DetailLevel.MEDIUM,
                        OutputFormat.PHASES_TABLE, _GROWING_TURNS),
    "pivot": Scenario("pivot", ProjectType.MOBILE_APP, DetailLevel.MEDIUM,
                      OutputFormat.PHASES_TABLE, _PIVOT_TURNS),
    "contradiction": Scenario("contradiction", ProjectType.INTERNAL_TOOL, DetailLevel.MEDIUM,
                              OutputFormat.PHASES_TABLE, _CONTRADICTION_TURNS),
}


def get_scenario(name: str) -> Scenario:
    try:
        return _SCENARIOS[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario {name!r}; available: {sorted(_SCENARIOS)}") from exc


def all_scenarios() -> list[Scenario]:
    """The three scenarios in deterministic order."""
    return [_SCENARIOS["growing"], _SCENARIOS["pivot"], _SCENARIOS["contradiction"]]
