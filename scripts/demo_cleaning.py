"""Demo: cleaning + Pandera validation over the seed budgets.

Loads every budget JSON from the seed corpus, runs ``clean_budget_records`` then
``validate_with_policy`` and prints the partition report. Expected story:

* 6 files go in, 5 reach validation (dedup collapses BUDGET-2024-0005).
* ``total_amount: -50000`` is discarded; the rest pass, including the
  ``12/03/2024`` date that sits next to ISO dates.

Usage::

    uv run python scripts/demo_cleaning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make ``app`` importable when this script runs directly from ``scripts/``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.cleaning import clean_budget_records, validate_with_policy  # noqa: E402

BUDGETS_DIR = ROOT / "data" / "seed" / "budgets"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    paths = sorted(BUDGETS_DIR.glob("*.json"))
    if not paths:
        print(f"ERROR: no budget JSON files under {BUDGETS_DIR}")
        return 1

    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    print(f"Files read: {len(records)}  ({BUDGETS_DIR})")

    df = clean_budget_records(records)
    print(f"Rows after cleaning + dedup: {len(df)}")

    result = validate_with_policy(df)
    print("\nReport:")
    for key, value in result.report.items():
        print(f"  {key}: {value}")

    if not result.discarded.empty:
        print("\nDiscarded:")
        print(result.discarded[["budget_id", "total_amount"]].to_string(index=False))
    if not result.quarantined.empty:
        print("\nQuarantined:")
        print(result.quarantined[["budget_id", "client_name"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
