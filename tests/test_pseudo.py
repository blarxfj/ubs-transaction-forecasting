from __future__ import annotations

import numpy as np
import pandas as pd

from ubs_forecasting.features import CUTOFF
from ubs_forecasting.model import FAMILY_ONLY_TARGET, candidate_rows, listwise_targets, softmax_objective
from ubs_forecasting.posterior import learn_amount_prior
from ubs_forecasting.pseudo import pseudo_labels, shift_history
from ubs_forecasting.vocab import CLASSES, FAMILIES


def _history() -> pd.DataFrame:
    rows = []

    def add(client: str, day: int, amount: float, description: str, mcc: str, kind: str = "card_payment") -> None:
        timestamp = (CUTOFF + pd.Timedelta(days=day)).tz_localize("UTC") + pd.Timedelta(hours=3)
        rows.append(
            {
                "client_id": client,
                "timestamp": timestamp,
                "amount": amount,
                "currency": "eur",
                "direction": "out",
                "type": kind,
                "mcc": mcc,
                "description": description,
                "fee": 0.0,
            }
        )

    # A: monthly gym stream running through the pseudo horizon.
    for day in range(-360, 0, 30):
        add("A", day, 29.9, "urban gym", "7997")
    # B: monthly cloud stream that stopped before the pseudo cutoff; one everyday purchase after it.
    for day in range(-360, -120, 30):
        add("B", day, 9.99, "cloud backup", "5732")
    add("B", -40, 55.0, "grocery store", "5411")
    # C: a single one-off subscription-like payment inside the horizon (not recurring).
    add("C", -200, 12.0, "coffee shop", "5812")
    add("C", -50, 14.5, "media streaming", "5812")
    frame = pd.DataFrame(rows)
    frame["date"] = frame.timestamp.dt.tz_localize(None).dt.floor("D")
    return frame


def _prior():
    unlabeled = _history()
    return learn_amount_prior(unlabeled)


def test_shift_history_drops_the_observed_future_and_redates_the_past() -> None:
    frame = _history()
    shifted = shift_history(frame, 90)
    assert shifted.date.max() < CUTOFF
    assert (shifted.date >= frame.date.min() + pd.Timedelta(days=90)).all()
    original_past = frame[frame.date < CUTOFF - pd.Timedelta(days=90)]
    assert len(shifted) == len(original_past)
    assert (shifted.date.values == (original_past.date + pd.Timedelta(days=90)).values).all()


def test_pseudo_labels_follow_the_earliest_recurring_event_in_the_horizon() -> None:
    labels = pseudo_labels(_history(), _prior(), 90)
    assert list(labels.columns) == list(CLASSES)
    assert np.allclose(labels.sum(axis=1), 1.0)
    assert labels.loc["A"].idxmax() == "gym"
    assert labels.loc["B", "none"] == 1.0
    assert labels.loc["C", "none"] == 1.0


def test_family_only_targets_mark_the_none_row_and_renormalize() -> None:
    features = pd.DataFrame(
        {
            "client_id": ["X"] * len(FAMILIES),
            "family": list(FAMILIES),
            "family_id": list(range(len(FAMILIES))),
            "p_prob": 0.5,
            "p_n": 2.0,
            "p_next": 3.0,
            "n_streams": 1.0,
            "n_active": 1.0,
            "c_rows": 10,
        }
    )
    rows = candidate_rows(features)
    soft = pd.DataFrame([[0.3, 0.3, 0, 0, 0, 0, 0, 0.4]], index=["X"], columns=list(CLASSES))
    target = listwise_targets(rows, soft, family_only=True).reshape(-1, len(CLASSES))
    assert target[0, -1] == FAMILY_ONLY_TARGET
    assert np.allclose(target[0, :2], 0.5)
    hard = listwise_targets(rows, pd.Series({"X": "gym"}))
    assert hard.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_family_only_blocks_give_no_gradient_to_the_none_row_and_weights_scale() -> None:
    raw = np.linspace(-1, 1, 2 * len(CLASSES))
    target = np.zeros((2, len(CLASSES)))
    target[0, 0] = 1.0
    target[1, 1] = 1.0
    target[1, -1] = FAMILY_ONLY_TARGET
    gradient, hessian = softmax_objective(target.reshape(-1), raw)
    gradient = gradient.reshape(2, -1)
    assert gradient[1, -1] == 0.0
    assert abs(gradient[1, :-1].sum()) < 1e-9
    assert abs(gradient[0].sum()) < 1e-9
    weighted, weighted_hessian = softmax_objective(target.reshape(-1), raw, np.full(2 * len(CLASSES), 2.0))
    assert np.allclose(weighted, 2 * gradient.reshape(-1))
    assert np.allclose(weighted_hessian, 2 * hessian)
