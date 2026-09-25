"""Per-(client, family) LightGBM ranker and client-level none gate."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .posterior import AmountPrior
from .vocab import CLASSES, FAMILIES

DEFAULT_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 4,
}

TrainingTable = tuple[pd.DataFrame, pd.Series]


@dataclass
class ModelBundle:
    """Serializable ranker/gate ensemble plus its preprocessing contract."""

    rankers: list[lgb.LGBMClassifier]
    none_gates: list[lgb.LGBMClassifier]
    ranker_columns: list[str]
    wide_columns: list[str]
    dropped_features: tuple[str, ...]
    seeds: tuple[int, ...]
    amount_prior: AmountPrior | None = None


def wide_features(long: pd.DataFrame, drop: Sequence[str] = ()) -> pd.DataFrame:
    """Pivot family features for the client-level none gate."""

    columns = [
        column
        for column in long.columns
        if not column.startswith("c_") and column not in ("client_id", "family", "family_id") and column not in drop
    ]
    wide = long.pivot(index="client_id", columns="family", values=columns)
    wide.columns = [f"{family}__{column}" for column, family in wide.columns]
    client_columns = [column for column in long.columns if column.startswith("c_")]
    return wide.join(long.groupby("client_id")[client_columns].first())


def fit_models(
    train_tables: Sequence[TrainingTable],
    *,
    amount_prior: AmountPrior | None = None,
    drop: Sequence[str] = (),
    seeds: Sequence[int] = (0, 1, 2),
    parameters: dict[str, Any] | None = None,
) -> ModelBundle:
    """Fit bagged family rankers and none gates."""

    if not train_tables:
        raise ValueError("at least one training table is required")
    long = pd.concat(
        [features.assign(_y=features.client_id.map(labels)) for features, labels in train_tables],
        ignore_index=True,
    )
    ranker_columns = [
        column for column in long.columns if column not in ("client_id", "family", "_y") and column not in drop
    ]
    ranker_rows = long[long._y != "none"]
    ranker_target = (ranker_rows.family == ranker_rows._y).astype(int)

    wide_tables = [wide_features(features, drop) for features, _ in train_tables]
    wide = pd.concat([table.reset_index(drop=True) for table in wide_tables], ignore_index=True)
    none_target = np.concatenate(
        [
            (labels.reindex(table.index) == "none").astype(int).values
            for table, (_, labels) in zip(wide_tables, train_tables, strict=True)
        ]
    )
    model_parameters = DEFAULT_PARAMETERS if parameters is None else parameters
    rankers: list[lgb.LGBMClassifier] = []
    none_gates: list[lgb.LGBMClassifier] = []
    for seed in seeds:
        seeded_parameters = {**model_parameters, "random_state": seed}
        ranker = lgb.LGBMClassifier(**seeded_parameters).fit(
            ranker_rows[ranker_columns],
            ranker_target,
            categorical_feature=["family_id"],
        )
        none_gate = lgb.LGBMClassifier(**seeded_parameters).fit(wide, none_target)
        rankers.append(ranker)
        none_gates.append(none_gate)
    return ModelBundle(
        rankers=rankers,
        none_gates=none_gates,
        ranker_columns=ranker_columns,
        wide_columns=list(wide.columns),
        dropped_features=tuple(drop),
        seeds=tuple(seeds),
        amount_prior=amount_prior,
    )


def predict_probabilities(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Combine normalized family scores with the independent none probability."""

    output: pd.DataFrame | int = 0
    for ranker, none_gate in zip(bundle.rankers, bundle.none_gates, strict=True):
        scores = pd.Series(
            ranker.predict_proba(features[bundle.ranker_columns])[:, 1],
            index=features.index,
        )
        probabilities = features.assign(score=scores).pivot(index="client_id", columns="family", values="score")[
            list(FAMILIES)
        ]
        probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
        wide = wide_features(features, bundle.dropped_features).reindex(columns=bundle.wide_columns)
        none_probability = pd.Series(none_gate.predict_proba(wide)[:, 1], index=wide.index).reindex(probabilities.index)
        probabilities = probabilities.mul(1 - none_probability, axis=0)
        probabilities["none"] = none_probability
        output = output + probabilities[list(CLASSES)] / len(bundle.seeds)
    assert isinstance(output, pd.DataFrame)
    return output


def fit_predict(
    train_tables: Sequence[TrainingTable],
    eval_tables: Sequence[pd.DataFrame],
    *,
    drop: Sequence[str] = (),
    seeds: Sequence[int] = (0, 1, 2),
    parameters: dict[str, Any] | None = None,
) -> list[pd.DataFrame]:
    """Convenience function for evaluation folds."""

    bundle = fit_models(train_tables, drop=drop, seeds=seeds, parameters=parameters)
    return [predict_probabilities(bundle, features) for features in eval_tables]


def decide(probabilities: pd.DataFrame) -> pd.Series:
    """Apply the untuned eight-way argmax decision rule."""

    labels = np.array(CLASSES)[np.argmax(probabilities.values, axis=1)]
    return pd.Series(labels, index=probabilities.index, name="prediction")


def save_model(bundle: ModelBundle, path: str | Path) -> None:
    """Serialize a trained ensemble."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=5)


def load_model(path: str | Path) -> ModelBundle:
    """Load a trained ensemble and validate its type."""

    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)  # noqa: S301 - local trusted model artifact
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"not a UBS forecasting model bundle: {path}")
    return bundle


# ----------------------------------------------------------------------------- listwise softmax

NONE_FAMILY_ID = len(FAMILIES)
SUMMARY_FEATURES = ("x_max_prob", "x_max_n", "x_min_next", "x_n_fam", "x_max_active")


def candidate_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Return eight contiguous candidate rows per client: seven families plus ``none``.

    The ``none`` row carries only the client-level ``c_*`` features. Every row additionally receives
    label-free client summaries of the family candidates so the shared scorer can compare a row
    against the strongest competing evidence of the same client.
    """

    client_columns = [column for column in features.columns if column.startswith("c_")]
    none_rows = features.groupby("client_id", sort=True)[client_columns].first().reset_index()
    none_rows["family"] = "none"
    none_rows["family_id"] = NONE_FAMILY_ID
    summary = features.groupby("client_id").agg(
        x_max_prob=("p_prob", "max"),
        x_max_n=("p_n", "max"),
        x_min_next=("p_next", "min"),
        x_n_fam=("n_streams", lambda values: float((values > 0).sum())),
        x_max_active=("n_active", "max"),
    )
    long = pd.concat([features, none_rows], ignore_index=True).join(summary, on="client_id")
    long = long.sort_values(["client_id", "family_id"], kind="stable").reset_index(drop=True)
    if len(long) % len(CLASSES):
        raise ValueError("candidate rows are not a multiple of the class count")
    return long


def softmax_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-client softmax cross-entropy over eight contiguous candidate rows."""

    raw = y_pred.reshape(-1, len(CLASSES))
    probabilities = np.exp(raw - raw.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gradient = (probabilities - y_true.reshape(-1, len(CLASSES))).reshape(-1)
    hessian = (probabilities * (1 - probabilities) + 1e-6).reshape(-1)
    return gradient, hessian


@dataclass
class ListwiseBundle:
    """Bagged listwise scorers plus their preprocessing contract."""

    models: list[lgb.LGBMRegressor]
    columns: list[str]
    dropped_features: tuple[str, ...]
    seeds: tuple[int, ...]
    amount_prior: AmountPrior | None = None


def fit_listwise(
    train_tables: Sequence[TrainingTable],
    *,
    amount_prior: AmountPrior | None = None,
    drop: Sequence[str] = (),
    seeds: Sequence[int] = (0, 1, 2),
    parameters: dict[str, Any] | None = None,
) -> ListwiseBundle:
    """Fit bagged listwise softmax scorers over eight candidate rows per client."""

    if not train_tables:
        raise ValueError("at least one training table is required")
    parts = []
    for features, labels in train_tables:
        rows = candidate_rows(features)
        parts.append(rows.assign(_y=rows.client_id.map(labels)))
    long = pd.concat(parts, ignore_index=True)
    columns = [column for column in long.columns if column not in ("client_id", "family", "_y") and column not in drop]
    target = (long.family == long._y).astype(int).to_numpy()
    model_parameters = {**(DEFAULT_PARAMETERS if parameters is None else parameters)}
    model_parameters.pop("objective", None)
    models = []
    for seed in seeds:
        model = lgb.LGBMRegressor(**model_parameters, objective=softmax_objective, random_state=seed)
        model.fit(long[columns], target, categorical_feature=["family_id"])
        models.append(model)
    return ListwiseBundle(
        models=models, columns=columns, dropped_features=tuple(drop), seeds=tuple(seeds), amount_prior=amount_prior
    )


def predict_listwise(bundle: ListwiseBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Average the per-client softmax over bagged scorers."""

    long = candidate_rows(features)
    output: pd.DataFrame | int = 0
    for model in bundle.models:
        raw = model.predict(long[bundle.columns]).reshape(-1, len(CLASSES))
        probabilities = np.exp(raw - raw.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        frame = pd.DataFrame(probabilities, index=long.client_id.to_numpy()[:: len(CLASSES)], columns=list(CLASSES))
        output = output + frame / len(bundle.models)
    assert isinstance(output, pd.DataFrame)
    output.index.name = "client_id"
    return output
