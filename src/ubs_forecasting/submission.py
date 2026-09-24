"""Submission creation and challenge-contract validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import decide
from .vocab import CLASSES

SUBMISSION_COLUMNS = ("client_id", "predicted_next_recurring_merchant")


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a submission violates the documented contract."""

    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns must be exactly {SUBMISSION_COLUMNS}")
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError("sample submission has an unexpected schema")
    if len(submission) != len(sample):
        raise ValueError("submission row count does not match sample submission")
    if submission.client_id.duplicated().any():
        raise ValueError("submission contains duplicate client IDs")
    if submission.client_id.tolist() != sample.client_id.tolist():
        raise ValueError("submission client IDs or ordering do not exactly match the sample")
    if submission.predicted_next_recurring_merchant.isna().any():
        raise ValueError("submission contains missing predictions")
    invalid = set(submission.predicted_next_recurring_merchant) - set(CLASSES)
    if invalid:
        raise ValueError(f"submission contains invalid labels: {sorted(invalid)}")


def create_submission(probabilities: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Create a contract-checked submission in sample order."""

    predictions = decide(probabilities).reindex(sample.client_id)
    submission = pd.DataFrame(
        {
            "client_id": sample.client_id.values,
            "predicted_next_recurring_merchant": predictions.values,
        }
    )
    validate_submission(submission, sample)
    return submission


def write_prediction_artifacts(
    probabilities: pd.DataFrame,
    sample_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write a submission and auditable class probabilities."""

    sample = pd.read_csv(sample_path, dtype={"client_id": str})
    submission = create_submission(probabilities, sample)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    probabilities.to_csv(output / "test_probabilities.csv")
    return submission
