"""Regression tests for hash partitioning and repeated client uncertainty."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from ubs_forecasting.listwise import softmax_objective
from ubs_forecasting.protocol import (
    client_folds,
    f1_from_counts,
    hash_mod,
    lockbox_ids,
    paired_repeated_difference,
    repeated_metrics,
)
from ubs_forecasting.vocab import CLASSES


def test_hash_uses_full_digest_and_only_valid_clients():
    train = pd.Index([f"T{i:03}" for i in range(100)])
    valid = pd.Index([f"V{i:03}" for i in range(100)])
    locked = lockbox_ids(valid)
    assert set(locked) <= set(valid)
    assert not set(locked) & set(train)
    assert all(hash_mod(client) == 0 for client in locked)
    for seed in (0, 1, 2):
        folds = client_folds(train.append(valid.drop(locked)), seed)
        expected = [
            int(hashlib.sha256(f"{seed}:{c}".encode()).hexdigest(), 16) % 5 for c in train.append(valid.drop(locked))
        ]
        np.testing.assert_array_equal(folds, expected)
    dev = train.append(valid.drop(locked))
    assert set(train) <= set(dev)


def test_fixed_labels_and_repeated_bootstrap():
    ids = pd.Index(["a", "b", "c", "d"])
    truth = pd.Series(["cloud", "gym", "gym", "none"], index=ids)
    prediction = np.array([0, 1, 0, 7])
    p = pd.DataFrame(np.eye(8)[prediction], columns=CLASSES, index=ids)
    expected = f1_score(truth, np.array(CLASSES)[prediction], labels=CLASSES, average="macro", zero_division=0)
    one = repeated_metrics(truth, [p], samples=100)
    repeated = repeated_metrics(truth, [p, p, p], samples=100)
    assert one["macro_f1"] == pytest.approx(expected)
    np.testing.assert_allclose(one["ci95"], repeated["ci95"])
    assert repeated["n_clients"] == 4
    paired = paired_repeated_difference(truth, [p, p, p], [p, p, p], samples=100)
    assert paired == {"difference": 0.0, "ci95": [0.0, 0.0]}
    with pytest.raises(ValueError, match="probabilities"):
        repeated_metrics(truth, [p.drop("a")])
    assert f1_from_counts(np.zeros((8, 8))) == 0


def test_listwise_gradient_is_group_normalized():
    target = np.eye(7)[[2, 4]].ravel()
    gradient, hessian = softmax_objective(target, np.zeros(14))
    np.testing.assert_allclose(gradient.reshape(-1, 7).sum(1), 0, atol=1e-12)
    assert np.all(hessian > 0)
    assert gradient[2] < 0 and gradient[11] < 0
    # The objective is invariant to an arbitrary offset within either group.
    shifted, _ = softmax_objective(target, np.repeat([2.5, -4.0], 7))
    np.testing.assert_allclose(gradient, shifted)
