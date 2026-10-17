"""Tests for the cleaning layer: dirt families, dedup policy and validation routing."""

from __future__ import annotations

import pandas as pd

from app.ingestion.cleaning import clean_budget_records, validate_with_policy


def _record(**overrides) -> dict:
    return {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme",
        "client_code": "CLI-0001",
        "currency": "EUR",
        "total_amount": 100.0,
        "signed_at": "2024-01-01",
        **overrides,
    }


# --- clean_budget_records ---------------------------------------------------


def test_disguised_nulls_become_na() -> None:
    df = clean_budget_records([_record(client_name="TBD")])

    assert pd.isna(df.loc[0, "client_name"])


def test_currency_casing_collapses_to_upper() -> None:
    df = clean_budget_records([_record(currency="eur")])

    assert df.loc[0, "currency"] == "EUR"


def test_date_coercion_handles_spanish_format() -> None:
    df = clean_budget_records([_record(signed_at="12/03/2024")])

    parsed = df.loc[0, "signed_at"]
    assert (parsed.year, parsed.month, parsed.day) == (2024, 3, 12)


def test_numeric_coercion_handles_strings() -> None:
    df = clean_budget_records([_record(total_amount="not-a-number")])

    assert pd.isna(df.loc[0, "total_amount"])


def test_dedup_keeps_latest_signed_at() -> None:
    df = clean_budget_records(
        [
            _record(budget_id="BUDGET-2024-0005", total_amount=30000, signed_at="2024-04-10"),
            _record(budget_id="BUDGET-2024-0005", total_amount=32000, signed_at="2024-04-12"),
        ]
    )

    assert len(df) == 1
    assert int(df.loc[0, "total_amount"]) == 32000


def test_empty_input_returns_empty_frame() -> None:
    assert clean_budget_records([]).empty


# --- validate_with_policy ---------------------------------------------------


def test_clean_valid_records_pass() -> None:
    result = validate_with_policy(clean_budget_records([_record()]))

    assert len(result.valid) == 1
    assert result.report["valid_rows"] == 1
    assert result.report["failures_by_check"] == {}


def test_integer_amounts_are_valid_after_cleaning() -> None:
    """JSON amounts such as 45000 reach the schema as int64; they must still validate."""
    result = validate_with_policy(clean_budget_records([_record(total_amount=45000)]))

    assert len(result.valid) == 1


def test_missing_client_name_is_let_through() -> None:
    # client_name is nullable in the schema, so a TBD name is valid, not quarantined.
    result = validate_with_policy(clean_budget_records([_record(client_name="TBD")]))

    assert len(result.valid) == 1
    assert result.report["failures_by_check"] == {}


def test_negative_amount_is_discarded() -> None:
    result = validate_with_policy(clean_budget_records([_record(total_amount=-5.0)]))

    assert len(result.discarded) == 1
    assert len(result.valid) == 0


def test_unknown_currency_is_discarded() -> None:
    result = validate_with_policy(clean_budget_records([_record(currency="JPY")]))

    assert len(result.discarded) == 1


def test_malformed_budget_id_is_discarded() -> None:
    result = validate_with_policy(clean_budget_records([_record(budget_id="BAD-1")]))

    assert len(result.discarded) == 1


def test_rows_are_partitioned_independently() -> None:
    df = clean_budget_records(
        [
            _record(budget_id="BUDGET-2024-0001"),
            _record(budget_id="BUDGET-2024-0002", client_code="bad"),
            _record(budget_id="BUDGET-2024-0003", total_amount=-1.0),
        ]
    )

    result = validate_with_policy(df)

    assert result.report["input_rows"] == 3
    assert result.report["valid_rows"] == 1
    assert result.report["quarantined_rows"] == 0
    assert result.report["discarded_rows"] == 2


def test_missing_required_column_quarantines_every_row() -> None:
    df = clean_budget_records([_record(), _record(budget_id="BUDGET-2024-0002")])

    result = validate_with_policy(df.drop(columns=["client_name"]))

    assert len(result.quarantined) == 2
    assert len(result.valid) == 0
    assert result.report["failures_by_check"] == {"column_in_dataframe": 1}


def test_empty_frame_short_circuits() -> None:
    result = validate_with_policy(pd.DataFrame())

    assert result.report == {"input_rows": 0}
