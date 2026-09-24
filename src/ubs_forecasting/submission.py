"""Submission creation and challenge-contract validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import decide
from .vocab import CLASSES

SUBMISSION_COLUMNS = ("client_id", "predicted_next_recurring_merchant")


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a submission violates the literal documented contract."""

    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns must be exactly {SUBMISSION_COLUMNS}")
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError("sample submission has an unexpected schema")
    if len(submission) != len(sample):
        raise ValueError("submission row count does not match sample submission")
    if submission.client_id.isna().any() or submission.client_id.duplicated().any():
        raise ValueError("submission contains null or duplicate client IDs")
    if set(submission.client_id) != set(sample.client_id):
        raise ValueError("submission client IDs do not exactly match the sample")
    if submission.predicted_next_recurring_merchant.isna().any():
        raise ValueError("submission contains missing predictions")
    invalid = set(submission.predicted_next_recurring_merchant) - set(CLASSES)
    if invalid:
        raise ValueError(f"submission contains invalid labels: {sorted(invalid)}")


def create_submission(probabilities: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Create a contract-checked submission in the sample's convenient order."""

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
    """Write and read back a submission, then write auditable class probabilities."""

    sample = pd.read_csv(sample_path, dtype={"client_id": str})
    submission = create_submission(probabilities, sample)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    read_back = pd.read_csv(submission_path, dtype={"client_id": str})
    validate_submission(read_back, sample)
    pd.testing.assert_frame_equal(read_back, submission, check_dtype=False)
    probabilities.to_csv(output / "test_probabilities.csv")
    return read_back
