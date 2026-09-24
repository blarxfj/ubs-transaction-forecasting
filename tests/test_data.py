from __future__ import annotations

import pandas as pd
import pytest

from ubs_forecasting.data import validate_transaction_frame


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["C1"],
            "timestamp": pd.to_datetime(["2025-12-31T00:00:00Z"], utc=True),
            "amount": [10.0],
            "currency": ["chf"],
            "direction": ["out"],
            "type": ["card_payment"],
            "mcc": ["5732"],
            "description": ["cloud access"],
            "fee": [0.0],
        }
    )


def test_input_invariants_accept_valid_history() -> None:
    validate_transaction_frame(_transactions(), "tiny")


def test_input_invariants_reject_future_history() -> None:
    frame = _transactions()
    frame.loc[0, "timestamp"] = pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(ValueError, match="cutoff"):
        validate_transaction_frame(frame, "tiny")


def test_input_invariants_reject_nonpositive_amount_and_schema_drift() -> None:
    frame = _transactions()
    frame.loc[0, "amount"] = 0
    with pytest.raises(ValueError, match="amounts"):
        validate_transaction_frame(frame, "tiny")

    frame = _transactions().drop(columns="fee")
    with pytest.raises(ValueError, match="columns"):
        validate_transaction_frame(frame, "tiny")
