"""Shared evaluation protocol: valid-only lockbox, hashed client folds, and bootstrap scoring.

The protocol is fixed so that every solution can be compared on identical client splits:

- A valid client is locked away when ``int(sha256(client_id), 16) % 5 == 0``.
- The development set is every train client plus every non-lockbox valid client.
- Fold assignment for a seed is ``int(sha256(f"{seed}:{client_id}"), 16) % 5``.
- Macro-F1 is always computed over the fixed eight labels.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .vocab import CLASSES

FOLDS = 5
FOLD_SEEDS = (0, 1, 2)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 1729


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def is_lockbox(client_id: str) -> bool:
    """Return whether a *valid* client belongs to the untouched lockbox."""

    return _hash(client_id) % FOLDS == 0


def fold_of(seed: int, client_id: str) -> int:
    """Return the deterministic fold of a client for one repetition seed."""

    return _hash(f"{seed}:{client_id}") % FOLDS


def split_development(train_labels: pd.Series, valid_labels: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return development labels, an is-valid flag over them, and lockbox labels."""

    locked = valid_labels.index.map(is_lockbox).values.astype(bool)
    lockbox = valid_labels[locked]
    development = pd.concat([train_labels, valid_labels[~locked]])
    if development.index.duplicated().any():
        raise ValueError("train and valid client identifiers overlap")
    is_valid = pd.Series(development.index.isin(valid_labels.index), index=development.index)
    return development, is_valid, lockbox


def fold_series(seed: int, client_ids: Sequence[str]) -> pd.Series:
    """Fold assignment for every client under one repetition seed."""

    return pd.Series([fold_of(seed, client_id) for client_id in client_ids], index=pd.Index(client_ids))


def macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(truth, prediction, average="macro", labels=list(CLASSES), zero_division=0))


def per_class_f1(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = f1_score(truth, prediction, average=None, labels=list(CLASSES), zero_division=0)
    return {label: float(value) for label, value in zip(CLASSES, values, strict=True)}


def argmax_labels(probabilities: pd.DataFrame) -> pd.Series:
    """Argmax over the fixed class order; ties resolve to the earlier class."""

    values = probabilities[list(CLASSES)].to_numpy()
    return pd.Series(np.array(CLASSES)[values.argmax(axis=1)], index=probabilities.index)


def _weighted_macro_f1(truth: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    """Macro-F1 under integer resample weights, computed from a weighted confusion matrix."""

    index = {label: position for position, label in enumerate(CLASSES)}
    truth_codes = np.array([index[value] for value in truth])
    prediction_codes = np.array([index[value] for value in prediction])
    confusion = np.bincount(truth_codes * len(CLASSES) + prediction_codes, weights=weight, minlength=len(CLASSES) ** 2)
    confusion = confusion.reshape(len(CLASSES), len(CLASSES))
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    diagonal = np.diag(confusion)
    scores = np.divide(2 * diagonal, denominator, out=np.zeros(len(CLASSES)), where=denominator > 0)
    return float(scores.mean())


def bootstrap_mean_macro_f1(
    truth: np.ndarray,
    predictions: Sequence[np.ndarray],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """95% client-bootstrap interval of the mean over repetitions of macro-F1.

    The same client resample is applied to every repetition so repeated predictions of one
    client are never treated as independent clients.
    """

    generator = np.random.default_rng(seed)
    values = np.empty(samples)
    for sample_index in range(samples):
        weight = np.bincount(generator.integers(len(truth), size=len(truth)), minlength=len(truth)).astype(float)
        values[sample_index] = np.mean([_weighted_macro_f1(truth, prediction, weight) for prediction in predictions])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def score_repetitions(
    labels: pd.Series,
    probabilities_by_seed: Mapping[int, pd.DataFrame],
    is_valid: pd.Series,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Score pooled development and validation-only macro-F1 across repetition seeds."""

    truth = labels.to_numpy()
    valid_mask = is_valid.reindex(labels.index).to_numpy()
    pooled_predictions = []
    seeds: dict[str, Any] = {}
    for seed, probabilities in sorted(probabilities_by_seed.items()):
        prediction = argmax_labels(probabilities.reindex(labels.index)).to_numpy()
        folds = fold_series(seed, list(labels.index)).to_numpy()
        fold_scores = [macro_f1(truth[folds == fold], prediction[folds == fold]) for fold in range(FOLDS)]
        pooled_predictions.append(prediction)
        seeds[str(seed)] = {
            "pooled_macro_f1": macro_f1(truth, prediction),
            "valid_only_macro_f1": macro_f1(truth[valid_mask], prediction[valid_mask]),
            "train_only_macro_f1": macro_f1(truth[~valid_mask], prediction[~valid_mask]),
            "mean_fold_macro_f1": float(np.mean(fold_scores)),
            "fold_macro_f1": fold_scores,
            "per_class_f1": per_class_f1(truth, prediction),
            "valid_only_per_class_f1": per_class_f1(truth[valid_mask], prediction[valid_mask]),
            "accuracy": float((truth == prediction).mean()),
        }
    valid_predictions = [prediction[valid_mask] for prediction in pooled_predictions]
    return {
        "n_development": int(len(truth)),
        "n_valid_only": int(valid_mask.sum()),
        "seeds": seeds,
        "pooled_macro_f1": float(np.mean([value["pooled_macro_f1"] for value in seeds.values()])),
        "pooled_ci95": bootstrap_mean_macro_f1(truth, pooled_predictions, samples=bootstrap_samples),
        "valid_only_macro_f1": float(np.mean([value["valid_only_macro_f1"] for value in seeds.values()])),
        "valid_only_ci95": bootstrap_mean_macro_f1(truth[valid_mask], valid_predictions, samples=bootstrap_samples),
        "valid_only_per_class_f1": {
            label: float(np.mean([value["valid_only_per_class_f1"][label] for value in seeds.values()]))
            for label in CLASSES
        },
    }


def score_lockbox(
    lockbox_labels: pd.Series,
    probabilities: pd.DataFrame,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Score the one-shot lockbox with a client bootstrap interval."""

    truth = lockbox_labels.to_numpy()
    prediction = argmax_labels(probabilities.reindex(lockbox_labels.index)).to_numpy()
    return {
        "n_lockbox": int(len(truth)),
        "macro_f1": macro_f1(truth, prediction),
        "ci95": bootstrap_mean_macro_f1(truth, [prediction], samples=bootstrap_samples),
        "per_class_f1": per_class_f1(truth, prediction),
        "accuracy": float((truth == prediction).mean()),
    }


def read_probability_table(path: str) -> pd.DataFrame:
    """Read a probability CSV indexed by client with the eight class columns."""

    frame = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    missing = set(CLASSES) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks probability columns {sorted(missing)}")
    return frame[list(CLASSES)].astype(float)


def write_probability_table(path: str, probabilities: pd.DataFrame, folds: pd.Series | None = None) -> None:
    """Write a probability table with an optional fold column after the client identifier."""

    frame = probabilities[list(CLASSES)].copy()
    if folds is not None:
        frame.insert(0, "fold", folds.reindex(frame.index).astype(int).values)
    frame.index.name = "client_id"
    frame.to_csv(path, float_format="%.10g")
