"""Pandera v2 schema for the cleaned budget DataFrame.

Pandera v2 import path: ``import pandera.pandas as pa`` (v1 used ``import pandera as pa``).

The schema mixes per-column constraints; the phase-level data is not on the flat
record, so only what is visible there is checked.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

BUDGET_ID_PATTERN = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_PATTERN = r"^CLI-\d{4}$"

BudgetRecord: DataFrameSchema = DataFrameSchema(
    columns={
        "budget_id": Column(
            str,
            checks=Check.str_matches(BUDGET_ID_PATTERN),
            nullable=False,
            required=True,
        ),
        "client_name": Column(
            str,
            nullable=True,  # TBD/N/A become NA during cleaning and are let through
            required=True,
        ),
        "client_code": Column(
            str,
            checks=Check.str_matches(CLIENT_CODE_PATTERN),
            nullable=False,
            required=True,
        ),
        "currency": Column(
            str,
            checks=Check.isin(["EUR", "USD", "GBP"]),
            nullable=False,
            required=True,
        ),
        "total_amount": Column(
            float,
            checks=[Check.ge(0), Check.le(10_000_000)],
            nullable=False,
            required=True,
        ),
        "signed_at": Column(
            "datetime64[ns]",
            nullable=False,
            required=True,
        ),
        # Added by clean_budget_records; without it strict=True rejects every cleaned frame.
        "content_hash": Column(str, nullable=True, required=False),
        # Present in the budget JSON but not validated; declared so strict=True accepts them.
        "contact": Column(nullable=True, required=False),
        "contact_email": Column(nullable=True, required=False),
        "notes": Column(nullable=True, required=False),
        "phases": Column(nullable=True, required=False),
    },
    strict=True,
    coerce=False,
)
