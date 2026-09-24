"""Shared family softmax ranker with an independent none gate."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax

from .model import DEFAULT_PARAMETERS, wide_features
from .vocab import CLASSES, FAMILIES


def softmax_objective(target: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seven contiguous family rows constitute one client/view choice set."""
    probabilities = softmax(raw.reshape(-1, 7), axis=1)
    gradient = probabilities - target.reshape(-1, 7)
    hessian = 2 * probabilities * (1 - probabilities) + 1e-6
    return gradient.ravel(), hessian.ravel()


def fit_predict_listwise(training, evaluations, *, drop=(), seeds=(0, 1, 2), rounds=400):
    """Fit on non-none clients; normalize scores jointly within each choice set."""
    long = pd.concat(
        [frame.assign(_y=frame.client_id.map(labels), _view=i) for i, (frame, labels) in enumerate(training)],
        ignore_index=True,
    ).sort_values(["_view", "client_id", "family_id"])
    if long._y.isna().any():
        raise ValueError("training features without labels")
    columns = [c for c in long if c not in ("client_id", "family", "_y", "_view") and c not in drop]
    rank = long[long._y != "none"]
    target = (rank.family == rank._y).astype(int)
    if not np.all(target.to_numpy().reshape(-1, 7).sum(1) == 1):
        raise ValueError("each ranker choice set must contain exactly one positive")
    wides = [wide_features(frame, drop) for frame, _ in training]
    wide = pd.concat(wides)
    none = np.concatenate(
        [(labels.reindex(w.index) == "none").astype(int) for w, (_, labels) in zip(wides, training, strict=True)]
    )
    output = [0 for _ in evaluations]
    for seed in seeds:
        params = {**DEFAULT_PARAMETERS, "n_estimators": rounds, "n_jobs": 2, "random_state": seed}
        gate = lgb.LGBMClassifier(**params).fit(wide, none)
        params["objective"] = softmax_objective
        model = lgb.LGBMRegressor(**params).fit(rank[columns], target, categorical_feature=["family_id"])
        for i, frame in enumerate(evaluations):
            frame = frame.sort_values(["client_id", "family_id"])
            raw = model.predict(frame[columns]).reshape(-1, 7)
            ids = pd.Index(frame.client_id.drop_duplicates(), name="client_id")
            p = pd.DataFrame(softmax(raw, axis=1), index=ids, columns=FAMILIES)
            w = wide_features(frame, drop).reindex(index=ids, columns=wide.columns)
            n = gate.predict_proba(w)[:, 1]
            p = p.mul(1 - n, axis=0)
            p["none"] = n
            output[i] = output[i] + p[list(CLASSES)] / len(seeds)
    return output
