---
name: add-estimation-example
description: 'Use when adding or editing a reference estimation example in the CAG (Context Augmented Generation) knowledge base (context/examples.py) that grounds the LLM prompt. Triggers: "add example", "new estimation sample", "CAG context", "reference estimation".'
---

# Add Estimation Example (CAG knowledge base)

## When to Use
- Growing the set of reference estimations injected into the system prompt.
- Adjusting pricing/format assumptions used across all examples.

## Existing Pattern
`estimator/app/context/examples.py` holds `ESTIMATION_EXAMPLES: list[dict]`, each entry with:
- `meeting_summary`: prose description of the client's request/constraints.
- `estimation`: markdown string that MUST follow the exact structure enforced in `build_system_prompt()` (`estimator/app/services/llm_service.py`):
  - `##` project title heading
  - `### Task Breakdown` table with columns `Task | Hours | Cost (EUR)`
  - `### Totals` with total hours and total cost
  - `### Recommended Team`
  - `### Estimated Duration`

`format_examples_for_prompt()` concatenates all examples as `--- EXAMPLE N ---` blocks and injects them into the system prompt built by `build_system_prompt()`.

Pricing baseline used across all examples (must stay consistent unless intentionally changed): **62.50 EUR/hour** (500 EUR/day) developer rate, **50 EUR/hour** (400 EUR/day) designer rate.

## Procedure
1. Write a realistic `meeting_summary` describing the project scope, integrations, and constraints (2-6 sentences).
2. Write the `estimation` markdown following the exact section structure above — copy an existing example in `ESTIMATION_EXAMPLES` as a template.
3. Verify hours × rate roughly reconciles with the stated totals per task, and total cost matches total hours × blended rate.
4. Append the new dict to `ESTIMATION_EXAMPLES` in `app/context/examples.py`.
5. If you change the global rate assumptions, update the rates stated in `build_system_prompt()` too, so the prompt and examples stay consistent.
6. There is no automated test for prompt content — smoke-test via `POST /api/v1/estimate` with a real transcription and confirm the LLM output format matches.
