from __future__ import annotations

import numpy as np
import pandas as pd

from ubs_forecasting.features import row_evidence
from ubs_forecasting.streams import build_streams, match_refunds, stream_record
from ubs_forecasting.vocab import FAMILIES


def _flat_prior() -> dict[str, object]:
    bins = np.linspace(np.log(1.5), np.log(800), 41)
    return {
        "bins": bins,
        "logp": {family: np.zeros(len(bins) - 1) for family in FAMILIES},
    }


def test_single_linkage_keeps_drift_and_splits_separated_amounts() -> None:
    frame = pd.DataFrame(
        {
            "amount": [10.0, 10.3, 10.6, 11.2, 11.25, 11.3],
            "currency": ["chf"] * 6,
            "day": [-70, -56, -42, -70, -56, -42],
        }
    )
    streams = build_streams(frame)
    assert [len(stream) for stream in streams] == [3, 3]


def test_biweekly_period_and_refund_matching() -> None:
    transactions = pd.DataFrame(
        {
            "client_id": ["C1"] * 4,
            "timestamp": pd.to_datetime(["2025-11-06", "2025-11-20", "2025-12-04", "2025-12-18"]),
            "date": pd.to_datetime(["2025-11-06", "2025-11-20", "2025-12-04", "2025-12-18"]),
            "amount": [66.0, 66.2, 66.1, 66.3],
            "currency": ["chf"] * 4,
            "direction": ["out"] * 4,
            "type": ["card_payment"] * 4,
            "mcc": ["7997"] * 4,
            "description": ["urban gym"] * 4,
            "fee": [0.0] * 4,
        }
    )
    stream = row_evidence(transactions)
    refunds = pd.DataFrame(
        {
            "day": [-12, -30],
            "amount": [66.3, 90.0],
            "currency": ["chf", "chf"],
            **{f"d_{family}": [float(family == "gym"), 0.0] for family in FAMILIES},
        }
    )
    matched = match_refunds(stream, refunds)
    assert matched.day.tolist() == [-12]

    record = stream_record(stream, refunds, "chf", _flat_prior())
    assert record["per"] == 14.0
    assert record["nref"] == 1
    assert record["last_refunded"] == 1.0
    assert FAMILIES[int(np.argmax(record["post"]))] == "gym"
