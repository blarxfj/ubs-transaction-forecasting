"""Shared client-hash splits and client-clustered repeated-CV uncertainty."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .vocab import CLASSES


def hash_mod(value: str, modulus: int = 5) -> int:
    """Use the full SHA-256 integer, not a truncated prefix."""
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % modulus


def lockbox_ids(valid_ids: pd.Index) -> pd.Index:
    """Only validation clients are eligible for the lockbox."""
    return valid_ids[[hash_mod(str(client)) == 0 for client in valid_ids]]


def client_folds(ids: pd.Index, seed: int) -> np.ndarray:
    return np.array([hash_mod(f"{seed}:{client}") for client in ids])


def f1_from_counts(counts: np.ndarray) -> float:
    diagonal = np.diag(counts)
    denominator = counts.sum(0) + counts.sum(1)
    return float(np.divide(2 * diagonal, denominator, out=np.zeros(8), where=denominator > 0).mean())


def repeated_metrics(
    truth: pd.Series, probabilities: list[pd.DataFrame], *, samples: int = 2000, seed: int = 2026
) -> dict:
    """Mean seed F1; bootstrap each client jointly across all fold repetitions.

    Also report probability-ensemble F1 separately. Repeated client predictions
    are never treated as independent observations for confidence intervals.
    """
    y = truth.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    predictions = []
    for frame in probabilities:
        values = frame.reindex(index=truth.index, columns=CLASSES).to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
            raise ValueError("missing or invalid probabilities")
        predictions.append(values.argmax(1))
    if not predictions or len(y) == 0 or samples < 1:
        raise ValueError("nonempty clients, repetitions, and bootstrap samples are required")

    def score(prediction: np.ndarray, indices: np.ndarray) -> float:
        counts = np.bincount(8 * y[indices] + prediction[indices], minlength=64).reshape(8, 8)
        return f1_from_counts(counts)

    indices = np.arange(len(y))
    scores = [score(p, indices) for p in predictions]
    generator = np.random.default_rng(seed)
    boot = [
        np.mean([score(p, selected) for p in predictions])
        for selected in (generator.integers(0, len(y), len(y)) for _ in range(samples))
    ]
    ensemble = sum(frame.reindex(index=truth.index, columns=CLASSES) for frame in probabilities) / len(probabilities)
    return {
        "n_clients": len(y),
        "per_seed_f1": scores,
        "macro_f1": float(np.mean(scores)),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "probability_ensemble_f1": score(ensemble.to_numpy().argmax(1), indices),
    }


def paired_repeated_difference(
    truth: pd.Series,
    first: list[pd.DataFrame],
    second: list[pd.DataFrame],
    *,
    samples: int = 2000,
    seed: int = 2028,
) -> dict:
    """Bootstrap second-minus-first mean seed F1 with paired client resampling."""
    if len(first) != len(second) or not first:
        raise ValueError("both methods must have the same positive number of repetitions")
    y = truth.map({label: i for i, label in enumerate(CLASSES)}).to_numpy()
    a = [p.reindex(index=truth.index, columns=CLASSES).to_numpy().argmax(1) for p in first]
    b = [p.reindex(index=truth.index, columns=CLASSES).to_numpy().argmax(1) for p in second]

    def difference(indices):
        differences = []
        for pa, pb in zip(a, b, strict=True):
            ca = np.bincount(8 * y[indices] + pa[indices], minlength=64).reshape(8, 8)
            cb = np.bincount(8 * y[indices] + pb[indices], minlength=64).reshape(8, 8)
            differences.append(f1_from_counts(cb) - f1_from_counts(ca))
        return float(np.mean(differences))

    generator = np.random.default_rng(seed)
    bootstrap = [difference(generator.integers(0, len(y), len(y))) for _ in range(samples)]
    return {
        "difference": difference(np.arange(len(y))),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }
