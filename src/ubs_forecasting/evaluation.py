"""Deterministic metrics, rule baselines, and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .vocab import CLASSES


def classification_metrics(truth: pd.Series, predictions: pd.Series) -> dict[str, Any]:
    """Return macro-F1, accuracy, and fixed-order per-class F1."""

    truth = truth.reindex(predictions.index)
    per_class = f1_score(truth, predictions, average=None, labels=CLASSES, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(
                truth,
                predictions,
                average="macro",
                labels=CLASSES,
                zero_division=0,
            )
        ),
        "accuracy": float((truth == predictions).mean()),
        "per_class_f1": {label: float(value) for label, value in zip(CLASSES, per_class, strict=True)},
    }


def bootstrap_intervals(
    truth: pd.Series,
    predictions: pd.Series,
    *,
    samples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap clients and return percentile CIs for macro and per-class F1."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    truth = truth.reindex(predictions.index)
    truth_values = truth.to_numpy()
    prediction_values = predictions.to_numpy()
    generator = np.random.default_rng(seed)
    macro_scores = np.empty(samples)
    class_scores = np.empty((samples, len(CLASSES)))
    for sample_index in range(samples):
        indices = generator.integers(0, len(truth_values), len(truth_values))
        sampled_truth = truth_values[indices]
        sampled_predictions = prediction_values[indices]
        macro_scores[sample_index] = f1_score(
            sampled_truth,
            sampled_predictions,
            average="macro",
            labels=CLASSES,
            zero_division=0,
        )
        class_scores[sample_index] = f1_score(
            sampled_truth,
            sampled_predictions,
            average=None,
            labels=CLASSES,
            zero_division=0,
        )
    tail = (1 - confidence) / 2
    quantiles = [tail, 1 - tail]
    macro_bounds = np.quantile(macro_scores, quantiles)
    class_bounds = np.quantile(class_scores, quantiles, axis=0)
    return {
        "confidence": confidence,
        "samples": samples,
        "macro_f1": [float(macro_bounds[0]), float(macro_bounds[1])],
        "per_class_f1": {
            label: [float(class_bounds[0, index]), float(class_bounds[1, index])] for index, label in enumerate(CLASSES)
        },
    }


def paired_bootstrap_difference(
    truth: pd.Series,
    first: pd.Series,
    second: pd.Series,
    *,
    samples: int = 2000,
    seed: int = 2028,
) -> dict[str, float | list[float]]:
    """Bootstrap the paired macro-F1 difference ``second - first`` by client."""

    truth = truth.reindex(first.index)
    second = second.reindex(first.index)
    truth_values = truth.to_numpy()
    first_values = first.to_numpy()
    second_values = second.to_numpy()
    generator = np.random.default_rng(seed)
    differences = np.empty(samples)
    for sample_index in range(samples):
        indices = generator.integers(0, len(truth_values), len(truth_values))
        kwargs = {"average": "macro", "labels": CLASSES, "zero_division": 0}
        first_score = f1_score(truth_values[indices], first_values[indices], **kwargs)
        second_score = f1_score(truth_values[indices], second_values[indices], **kwargs)
        differences[sample_index] = second_score - first_score
    bounds = np.quantile(differences, [0.025, 0.975])
    point = f1_score(truth_values, second_values, average="macro", labels=CLASSES, zero_division=0) - f1_score(
        truth_values, first_values, average="macro", labels=CLASSES, zero_division=0
    )
    return {
        "difference": float(point),
        "interval": [float(bounds[0]), float(bounds[1])],
    }


def metric_report(
    truth: pd.Series,
    predictions: pd.Series,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Combine point estimates and bootstrap intervals."""

    return {
        **classification_metrics(truth, predictions),
        "bootstrap": bootstrap_intervals(
            truth,
            predictions,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def validate_leak_regression(
    leaky_macro_f1: float,
    safe_macro_f1: float,
    *,
    minimum_safe_score: float = 0.55,
    minimum_recovery: float = 0.15,
) -> None:
    """Guard the known train-only masking leak and the leak-safe fallback."""

    if safe_macro_f1 < minimum_safe_score:
        raise RuntimeError(f"leak-safe train-only model regressed: {safe_macro_f1:.3f} < {minimum_safe_score:.3f}")
    recovery = safe_macro_f1 - leaky_macro_f1
    if recovery < minimum_recovery:
        raise RuntimeError(
            f"expected masking-leak contrast is missing: recovery {recovery:.3f} < {minimum_recovery:.3f}"
        )


def none_auc(truth: pd.Series, probabilities: pd.DataFrame) -> float:
    """Calculate ROC AUC for the none gate."""

    return float(roc_auc_score(truth.reindex(probabilities.index) == "none", probabilities["none"]))


def format_metric_line(title: str, metrics: dict[str, Any]) -> str:
    """Format a stable one-line metric summary."""

    per_class = " ".join(f"{label}={metrics['per_class_f1'][label]:.3f}" for label in CLASSES)
    return f"{title:52s} macro-F1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} | {per_class}"


def recurrence_rule_predictions(long: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Series]:
    """Compare three deterministic recurrence-ranking heuristics."""

    active = long[(long.p_active == 1) & (long.p_prob >= 0.4)]
    singles = (
        long[(long.s_prob >= 0.5) & (long.s_last >= -45)]
        .sort_values("s_last", ascending=False)
        .groupby("client_id")
        .family.first()
    )
    rankings = {
        "earliest_next": active.sort_values("p_next").groupby("client_id").family.first(),
        "most_payments": (
            active.sort_values(["p_n", "p_next"], ascending=[False, True]).groupby("client_id").family.first()
        ),
        "most_recent": (
            active.sort_values(["p_last", "p_next"], ascending=[False, True]).groupby("client_id").family.first()
        ),
    }
    return {
        name: ranked.reindex(labels.index).fillna(singles.reindex(labels.index)).fillna("none")
        for name, ranked in rankings.items()
    }


def rule_predictions(long: pd.DataFrame, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the earliest-next rule and its refunded-stream none variant."""

    rule_one = recurrence_rule_predictions(long, labels)["earliest_next"]
    refunded = (
        long.groupby("client_id")[["c_n_active_ref", "c_n_ended_ref"]]
        .first()
        .sum(axis=1)
        .reindex(labels.index)
        .fillna(0)
    )
    rule_four = rule_one.copy()
    rule_four[refunded >= 2] = "none"
    return rule_one, rule_four
