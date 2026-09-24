from __future__ import annotations

import pandas as pd
import pytest

from ubs_forecasting.submission import create_submission, validate_submission
from ubs_forecasting.vocab import CLASSES


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["C2", "C1"],
            "predicted_next_recurring_merchant": ["none", "none"],
        }
    )


def test_submission_uses_exact_schema_ids_order_and_labels() -> None:
    probabilities = pd.DataFrame(
        [[0.9, *([0.0] * 7)], [0.0, *([0.0] * 6), 1.0]],
        index=["C1", "C2"],
        columns=CLASSES,
    )
    submission = create_submission(probabilities, _sample())
    assert submission.columns.tolist() == [
        "client_id",
        "predicted_next_recurring_merchant",
    ]
    assert submission.client_id.tolist() == ["C2", "C1"]
    assert submission.predicted_next_recurring_merchant.tolist() == ["none", "cloud"]


def test_submission_accepts_any_order_but_rejects_wrong_ids_and_labels() -> None:
    reordered = _sample().iloc[::-1].reset_index(drop=True)
    validate_submission(reordered, _sample())

    wrong_ids = _sample()
    wrong_ids.loc[0, "client_id"] = "C3"
    with pytest.raises(ValueError, match="client IDs"):
        validate_submission(wrong_ids, _sample())

    duplicate = _sample()
    duplicate.loc[0, "client_id"] = "C1"
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(duplicate, _sample())

    invalid = _sample()
    invalid.loc[0, "predicted_next_recurring_merchant"] = "merchant"
    with pytest.raises(ValueError, match="invalid labels"):
        validate_submission(invalid, _sample())
