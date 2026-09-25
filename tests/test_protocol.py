import hashlib

import numpy as np
import pandas as pd
import pytest

from ubs_forecasting.ensemble import blend
from ubs_forecasting.protocol import (
    argmax_labels,
    bootstrap_mean_macro_f1,
    fold_of,
    is_lockbox,
    macro_f1,
    score_repetitions,
    split_development,
)
from ubs_forecasting.vocab import CLASSES


def test_lockbox_and_fold_rules_match_the_documented_hashes():
    client_id = "C000005"
    digest = int(hashlib.sha256(client_id.encode()).hexdigest(), 16)
    assert is_lockbox(client_id) == (digest % 5 == 0)
    fold_digest = int(hashlib.sha256(f"2:{client_id}".encode()).hexdigest(), 16)
    assert fold_of(2, client_id) == fold_digest % 5


def test_split_development_keeps_every_train_client_and_locks_only_valid_clients():
    train = pd.Series(["none", "gym"], index=["T1", "T2"])
    valid_ids = [f"V{i:03d}" for i in range(50)]
    valid = pd.Series(["cloud"] * 50, index=valid_ids)
    development, is_valid, lockbox = split_development(train, valid)
    assert set(train.index) <= set(development.index)
    assert all(is_lockbox(client_id) for client_id in lockbox.index)
    assert not any(is_lockbox(client_id) for client_id in development.index if client_id.startswith("V"))
    assert is_valid.loc["T1"] is np.False_ or not is_valid.loc["T1"]
    assert len(development) + len(lockbox) == 52


def test_overlapping_identifiers_are_rejected():
    with pytest.raises(ValueError):
        split_development(pd.Series(["none"], index=["X"]), pd.Series(["gym"], index=["X"]))


def test_macro_f1_uses_all_fixed_labels():
    truth = np.array(["gym", "gym", "none"])
    prediction = np.array(["gym", "gym", "none"])
    # Perfect on two present classes, but six absent classes contribute zero F1.
    assert macro_f1(truth, prediction) == pytest.approx(2 / 8)


def test_bootstrap_interval_contains_the_point_estimate():
    generator = np.random.default_rng(0)
    truth = generator.choice(CLASSES, 300)
    prediction = np.where(generator.random(300) < 0.7, truth, generator.choice(CLASSES, 300))
    lower, upper = bootstrap_mean_macro_f1(truth, [prediction], samples=200)
    assert lower <= macro_f1(truth, prediction) <= upper


def test_score_repetitions_reports_pooled_and_valid_only_views():
    ids = [f"C{i:03d}" for i in range(40)]
    labels = pd.Series([CLASSES[i % 8] for i in range(40)], index=ids)
    is_valid = pd.Series([i >= 20 for i in range(40)], index=ids)
    probabilities = pd.DataFrame(np.eye(8)[[CLASSES.index(label) for label in labels]], index=ids, columns=CLASSES)
    result = score_repetitions(labels, {0: probabilities, 1: probabilities}, is_valid, bootstrap_samples=20)
    assert result["pooled_macro_f1"] == pytest.approx(1.0)
    assert result["valid_only_macro_f1"] == pytest.approx(1.0)
    assert result["n_valid_only"] == 20


def test_blend_normalizes_weights_and_ignores_missing_components():
    ids = ["A", "B"]
    columns = ["cloud", "none"]
    first = pd.DataFrame([[0.8, 0.2], [0.5, 0.5]], index=ids, columns=columns).reindex(columns=CLASSES).fillna(0)
    second = pd.DataFrame([[0.2, 0.8], [0.5, 0.5]], index=ids, columns=columns).reindex(columns=CLASSES).fillna(0)
    blended = blend({"x": first, "y": second}, {"x": 1.0, "y": 3.0, "absent": 5.0})
    assert blended.loc["A", "cloud"] == pytest.approx(0.35)
    assert list(argmax_labels(blended)) == ["none", "cloud"]
