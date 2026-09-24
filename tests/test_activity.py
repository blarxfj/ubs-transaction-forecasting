"""Activity covariates must not encode description quality or future events."""

import pandas as pd

from ubs_forecasting.activity import activity_features


def test_activity_is_noise_invariant():
    frame = pd.DataFrame(
        {
            "client_id": ["a"] * 4,
            "date": pd.to_datetime(["2025-12-30", "2025-12-15", "2025-11-20", "2025-10-01"]),
            "type": ["card_payment", "refund", "card_payment", "card_payment"],
            "direction": ["out", "in", "out", "out"],
            "description": ["cloud access"] * 4,
            "mcc": ["5732"] * 4,
        }
    )
    first = activity_features(frame)
    changed = activity_features(frame.assign(description="monthly plan", mcc="7997"))
    pd.testing.assert_frame_equal(first, changed)
    assert first.loc["a", "c_card_count30"] == 1
    assert first.loc["a", "c_refund_count30"] == 1
    assert first.loc["a", "c_last_activity"] == -2
    assert first.loc["a", "c_card_change30"] == 1
