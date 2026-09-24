"""Measured global and local explanations for the two-part LightGBM model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import ModelBundle, wide_features
from .vocab import FAMILIES


def feature_group(feature: str) -> str:
    """Map a model column to a human-readable signal group."""

    base = feature.split("__", 1)[-1]
    if any(token in base for token in ("gap_last", "gap_max", "skipped_cycle", "amt_change_last")):
        return "ordered_sequence"
    if any(token in base for token in ("ref", "refund")):
        return "refunds"
    if any(token in base for token in ("next", "last", "first", "span", "per", "overdue", "active", "due")):
        return "timing_recurrence"
    if any(token in base for token in ("prob", "mcc", "generic", "amt", "family_id", "odd_cur")):
        return "merchant_evidence"
    if base.startswith("c_") or "n_streams" in base:
        return "background_activity"
    return "other"


def gain_importance(bundle: ModelBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate LightGBM gain importance across ensemble members."""

    rows = []
    for component, models, columns in (
        ("family_ranker", bundle.rankers, bundle.ranker_columns),
        ("none_gate", bundle.none_gates, bundle.wide_columns),
    ):
        gains = np.mean(
            [model.booster_.feature_importance(importance_type="gain") for model in models],
            axis=0,
        )
        total = gains.sum()
        for feature, gain in zip(columns, gains, strict=True):
            rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "group": feature_group(feature),
                    "gain": float(gain),
                    "gain_share": float(gain / total) if total else 0.0,
                }
            )
    detailed = pd.DataFrame(rows).sort_values(["component", "gain_share"], ascending=[True, False])
    grouped = (
        detailed.groupby(["component", "group"], as_index=False)
        .gain_share.sum()
        .sort_values(["component", "gain_share"], ascending=[True, False])
    )
    return detailed, grouped


def _mean_contributions(models: list, features: pd.DataFrame) -> np.ndarray:
    return np.mean(
        [model.booster_.predict(features, pred_contrib=True) for model in models],
        axis=0,
    )


def local_explanations(
    bundle: ModelBundle,
    test_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Explain one high-none and one high-family test prediction with model contributions."""

    none_client = probabilities["none"].idxmax()
    family_scores = probabilities[list(FAMILIES)]
    family_client = family_scores.max(axis=1).idxmax()
    clients = ((none_client, "high_none"), (family_client, "high_family"))
    rows = []
    wide = wide_features(test_features, bundle.dropped_features).reindex(columns=bundle.wide_columns)
    for client_id, example in clients:
        none_row = wide.loc[[client_id]]
        none_contributions = _mean_contributions(bundle.none_gates, none_row)[0]
        none_pairs = list(zip([*bundle.wide_columns, "bias"], none_contributions, strict=True))
        for feature, contribution in sorted(none_pairs[:-1], key=lambda item: abs(item[1]), reverse=True)[:top_n]:
            rows.append(
                {
                    "client_id": client_id,
                    "example": example,
                    "component": "none_gate",
                    "predicted_class": probabilities.loc[client_id].idxmax(),
                    "feature": feature,
                    "feature_value": float(none_row.iloc[0][feature]),
                    "logit_contribution": float(contribution),
                }
            )

        family = family_scores.loc[client_id].idxmax()
        family_row = test_features[(test_features.client_id == client_id) & (test_features.family == family)][
            bundle.ranker_columns
        ]
        rank_contributions = _mean_contributions(bundle.rankers, family_row)[0]
        rank_pairs = list(zip([*bundle.ranker_columns, "bias"], rank_contributions, strict=True))
        for feature, contribution in sorted(rank_pairs[:-1], key=lambda item: abs(item[1]), reverse=True)[:top_n]:
            value = family_row.iloc[0][feature]
            rows.append(
                {
                    "client_id": client_id,
                    "example": example,
                    "component": f"family_ranker:{family}",
                    "predicted_class": probabilities.loc[client_id].idxmax(),
                    "feature": feature,
                    "feature_value": float(value) if pd.notna(value) else np.nan,
                    "logit_contribution": float(contribution),
                }
            )
    return pd.DataFrame(rows)


def write_interpretability_artifacts(
    bundle: ModelBundle,
    test_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write global gain importance and two local contribution examples."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detailed, grouped = gain_importance(bundle)
    detailed.to_csv(output / "feature_importance.csv", index=False)
    grouped.to_csv(output / "feature_group_importance.csv", index=False)
    local_explanations(bundle, test_features, probabilities).to_csv(output / "local_explanations.csv", index=False)
