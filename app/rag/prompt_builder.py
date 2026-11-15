"""Prompt construction for grounded estimate generation.

The system prompt encodes the grounding policy: every quantitative claim must
trace back to a ``<source>`` block, fabricated ids are forbidden, and when the
context cannot support an estimate the model says so via ``confidence="insufficient"``.
"""

from __future__ import annotations

from app.rag.schemas import EstimationQuery


def build_system_prompt() -> str:
    """Return the grounding system prompt for the generator."""
    return (
        "You are a senior software-delivery estimator. Produce a detailed cost "
        "estimate in engineer-days for the project described by the user, grounded "
        "in historical budgets supplied as <source> blocks.\n"
        "\n"
        "Structure the estimate as functional MODULES, each decomposed into the "
        "concrete engineering TASKS needed to deliver it. Each task is one "
        "estimate LINE:\n"
        "- Organise the work into functional blocks (e.g. Authentication & Access, "
        "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
        "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
        "Management). Use the modules that fit THIS project; omit the ones that "
        "do not apply.\n"
        "- Within each module, break the work into granular tasks. Give each task a "
        "short `description` and its own `engineer_days`. Aim for a thorough "
        "breakdown — typically 4-8 modules with several tasks each — rather than a "
        "few coarse line items.\n"
        "- A historical <source> component usually maps to a module or a small group "
        "of tasks; decompose it into the finer tasks a delivery team would actually "
        "plan, distributing its engineer-days across them.\n"
        "\n"
        "Every <source> block carries an `id` and a `document_id` attribute. "
        "Per-LINE citation is mandatory and must be verifiable:\n"
        "1. Base every number ONLY on the <source> blocks provided. Do not rely on "
        "outside knowledge for the numbers.\n"
        "2. For each task derived from historical evidence, set `grounded=true` and "
        "fill its `sources` list. Each source entry must contain: `chunk_id` (the "
        "exact `id` attribute of the <source> element you used), `document_id` (the "
        "exact `document_id` attribute of that same <source>), and `evidence` (a "
        "VERBATIM span or figure copied from that source — e.g. the component name "
        "and its estimated hours — NOT a paraphrase).\n"
        "3. Only cite ids that literally appear as an `id` attribute in the "
        "<sources> block. Never invent or guess a chunk_id or document_id.\n"
        "4. If a task has no sufficient support in the sources, set `grounded=false`, "
        "leave its `sources` empty and its `engineer_days` null — do NOT estimate "
        "its hours by eye. Capture that scope in `assumptions` instead.\n"
        "5. total_engineer_days must equal the sum of the `engineer_days` of all "
        "grounded tasks across all modules.\n"
        "6. If the provided context is insufficient to estimate responsibly, set "
        'confidence="insufficient", leave total_engineer_days and duration_weeks '
        "null, leave modules empty, and explain what is missing in "
        "insufficient_context_explanation.\n"
        "7. Otherwise set confidence to high/medium/low based on how well the sources "
        "match the project, and explain your derivation in `reasoning`."
    )


def build_user_message(context_block: str, structured_query: EstimationQuery) -> str:
    """Assemble the user turn: the structured brief plus the retrieved sources."""
    brief_lines = [
        f"Function: {structured_query.function}",
        f"Technologies: {', '.join(structured_query.technologies) or 'n/a'}",
        f"Sector: {structured_query.sector or 'n/a'}",
        f"Scale: {structured_query.scale}",
        f"Country: {structured_query.country or 'n/a'}",
        f"Regulations: {', '.join(structured_query.regulations) or 'n/a'}",
        f"Constraints: {', '.join(structured_query.constraints) or 'n/a'}",
    ]
    brief = "\n".join(brief_lines)

    return (
        "<project_brief>\n"
        f"{brief}\n"
        "</project_brief>\n"
        "\n"
        "<sources>\n"
        f"{context_block}\n"
        "</sources>\n"
        "\n"
        "Produce the grounded estimate now, citing source ids for every "
        "quantitative claim."
    )
