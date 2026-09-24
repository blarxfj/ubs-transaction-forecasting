from __future__ import annotations

from pathlib import Path

import pandas as pd

from ubs_forecasting.model import DEFAULT_PARAMETERS, fit_models, predict_probabilities
from ubs_forecasting.submission import write_prediction_artifacts
from ubs_forecasting.vocab import CLASSES, FAMILIES


def _synthetic_features(client_ids: list[str], labels: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for client_number, client_id in enumerate(client_ids):
        label = labels.loc[client_id] if labels is not None else None
        for family_id, family in enumerate(FAMILIES):
            rows.append(
                {
                    "client_id": client_id,
                    "family": family,
                    "family_id": family_id,
                    "p_next": float((family_id * 7 + client_number * 3) % 31),
                    "p_prob": 0.9 if family == label else 0.1 + family_id / 100,
                    "p_n": float(2 + (client_number + family_id) % 6),
                    "n_active": float((client_number + family_id) % 3),
                    "n_streams": float(1 + (client_number * family_id) % 4),
                    "e_ref": float((client_number + 2 * family_id) % 5),
                    "c_rows": 40 + client_number,
                    "c_n_active_ref": int(label == "none") if label is not None else client_number % 2,
                    "c_n_ended_ref": (client_number // 2) % 3,
                }
            )
    return pd.DataFrame(rows)


def _fit_and_write(output: Path, sample_path: Path) -> bytes:
    client_ids = [f"C{index:03d}" for index in range(32)]
    labels = pd.Series(
        [CLASSES[index % len(CLASSES)] for index in range(len(client_ids))],
        index=client_ids,
    )
    train_features = _synthetic_features(client_ids, labels)
    parameters = {
        **DEFAULT_PARAMETERS,
        "n_estimators": 20,
        "n_jobs": 1,
        "min_child_samples": 2,
    }
    bundle = fit_models(
        [(train_features, labels)],
        seeds=(0, 1),
        parameters=parameters,
    )
    test_ids = ["T002", "T000", "T001"]
    probabilities = predict_probabilities(bundle, _synthetic_features(test_ids))
    write_prediction_artifacts(probabilities, sample_path, output)
    return (output / "submission.csv").read_bytes()


def test_two_model_runs_produce_byte_identical_submission(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "client_id": ["T002", "T000", "T001"],
            "predicted_next_recurring_merchant": ["none"] * 3,
        }
    ).to_csv(sample_path, index=False)

    first = _fit_and_write(tmp_path / "first", sample_path)
    second = _fit_and_write(tmp_path / "second", sample_path)
    assert first == second
