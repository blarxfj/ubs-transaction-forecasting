from __future__ import annotations

import numpy as np
import pandas as pd
from test_determinism import _synthetic_features

from ubs_forecasting.model import DEFAULT_PARAMETERS, candidate_rows, fit_listwise, predict_listwise
from ubs_forecasting.vocab import CLASSES


def _labels(client_ids: list[str]) -> pd.Series:
    return pd.Series([CLASSES[index % len(CLASSES)] for index in range(len(client_ids))], index=client_ids)


def test_candidate_rows_add_one_none_row_per_client_in_contiguous_blocks() -> None:
    client_ids = [f"C{index:03d}" for index in range(5)]
    rows = candidate_rows(_synthetic_features(client_ids))
    assert len(rows) == 8 * len(client_ids)
    assert rows.family_id.tolist() == list(range(8)) * len(client_ids)
    assert rows.client_id.tolist() == sorted(rows.client_id.tolist())
    none_rows = rows[rows.family == "none"]
    assert none_rows.p_next.isna().all()
    assert (none_rows.x_max_n == rows[rows.family != "none"].groupby("client_id").p_n.max().values).all()


def test_listwise_probabilities_are_normalized_and_deterministic() -> None:
    client_ids = [f"C{index:03d}" for index in range(32)]
    labels = _labels(client_ids)
    parameters = {**DEFAULT_PARAMETERS, "n_estimators": 20, "n_jobs": 1, "min_child_samples": 2}
    outputs = []
    for _ in range(2):
        bundle = fit_listwise([(_synthetic_features(client_ids, labels), labels)], seeds=(0, 1), parameters=parameters)
        outputs.append(predict_listwise(bundle, _synthetic_features(["T002", "T000", "T001"])))
    first, second = outputs
    assert list(first.columns) == list(CLASSES)
    assert list(first.index) == ["T000", "T001", "T002"]
    assert np.allclose(first.sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(first, second)
