"""Dataset loading, input invariants, and reproducibility metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .vocab import CLASSES

TRANSACTION_SPLITS = ("unlabeled_pretrain", "train", "valid", "test")
TRANSACTION_COLUMNS = frozenset(
    {
        "client_id",
        "timestamp",
        "amount",
        "currency",
        "direction",
        "type",
        "mcc",
        "description",
        "fee",
    }
)
LABEL_COLUMNS = ("client_id", "cutoff_date", "target_next_recurring_merchant")
SUBMISSION_COLUMNS = ("client_id", "predicted_next_recurring_merchant")
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
REQUIRED_FILES = (
    "unlabeled_pretrain_transactions.jsonl",
    "train_transactions.jsonl",
    "train_labels.csv",
    "valid_transactions.jsonl",
    "valid_labels.csv",
    "test_transactions.jsonl",
    "sample_submission.csv",
)


def validate_data_directory(data_dir: str | Path) -> Path:
    """Validate and return a directory containing the complete challenge package."""

    path = Path(data_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required data files in {path}: {', '.join(missing)}")
    return path


def validate_transaction_frame(frame: pd.DataFrame, split: str) -> None:
    """Reject malformed, nonpositive, null, or post-cutoff transaction histories."""

    actual_columns = set(frame.columns) - {"date"}
    if actual_columns != TRANSACTION_COLUMNS:
        missing = sorted(TRANSACTION_COLUMNS - actual_columns)
        extra = sorted(actual_columns - TRANSACTION_COLUMNS)
        raise ValueError(f"{split} transaction columns differ; missing={missing}, extra={extra}")
    if frame.empty:
        raise ValueError(f"{split} transactions are empty")
    if frame[list(TRANSACTION_COLUMNS)].isna().any().any():
        raise ValueError(f"{split} transactions contain null values")
    if (frame.client_id.astype(str).str.len() == 0).any():
        raise ValueError(f"{split} transactions contain empty client IDs")
    if not np.isfinite(frame.amount).all() or (frame.amount <= 0).any():
        raise ValueError(f"{split} transactions contain nonpositive or nonfinite amounts")
    if not np.isfinite(frame.fee).all() or (frame.fee < 0).any():
        raise ValueError(f"{split} transactions contain negative or nonfinite fees")
    if not frame.direction.isin(["in", "out"]).all():
        raise ValueError(f"{split} transactions contain an unknown direction")
    if (frame.timestamp >= CUTOFF).any():
        raise ValueError(f"{split} contains a transaction on or after the cutoff")


def load_transactions(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load one transaction split, enforce invariants, and add normalized date fields."""

    if split not in TRANSACTION_SPLITS:
        raise ValueError(f"unknown split: {split}")
    path = Path(data_dir) / f"{split}_transactions.jsonl"
    frame = pd.read_json(path, lines=True, dtype={"mcc": str, "client_id": str})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    validate_transaction_frame(frame, split)
    frame["date"] = frame.timestamp.dt.tz_localize(None).dt.floor("D")
    return frame


def load_label_frame(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load and validate one labeled split."""

    if split not in {"train", "valid"}:
        raise ValueError("labels only exist for train and valid")
    path = Path(data_dir) / f"{split}_labels.csv"
    frame = pd.read_csv(path, dtype={"client_id": str})
    if tuple(frame.columns) != LABEL_COLUMNS:
        raise ValueError(f"{split} label columns must be exactly {LABEL_COLUMNS}")
    if frame.isna().any().any() or frame.client_id.duplicated().any():
        raise ValueError(f"{split} labels contain null or duplicate client IDs")
    if set(frame.cutoff_date.astype(str)) != {"2026-01-01"}:
        raise ValueError(f"{split} labels contain an unexpected cutoff")
    invalid = set(frame.target_next_recurring_merchant) - set(CLASSES)
    if invalid:
        raise ValueError(f"{split} labels contain invalid classes: {sorted(invalid)}")
    return frame


def load_labels(data_dir: str | Path, split: str) -> pd.Series:
    """Load labels indexed by client ID."""

    return load_label_frame(data_dir, split).set_index("client_id").target_next_recurring_merchant


def validate_dataset_relationships(data_dir: str | Path) -> None:
    """Check split isolation and exact label/sample client relationships."""

    path = validate_data_directory(data_dir)
    client_sets: dict[str, set[str]] = {}
    for split in TRANSACTION_SPLITS:
        frame = load_transactions(path, split)
        client_sets[split] = set(frame.client_id)
    for index, split in enumerate(TRANSACTION_SPLITS):
        for other in TRANSACTION_SPLITS[index + 1 :]:
            overlap = client_sets[split] & client_sets[other]
            if overlap:
                raise ValueError(f"client IDs overlap between {split} and {other}")
    for split in ("train", "valid"):
        labels = load_label_frame(path, split)
        if set(labels.client_id) != client_sets[split]:
            raise ValueError(f"{split} label IDs do not exactly match transaction IDs")
    sample = pd.read_csv(path / "sample_submission.csv", dtype={"client_id": str})
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"sample columns must be exactly {SUBMISSION_COLUMNS}")
    if sample.client_id.isna().any() or sample.client_id.duplicated().any():
        raise ValueError("sample submission contains null or duplicate client IDs")
    if set(sample.client_id) != client_sets["test"]:
        raise ValueError("sample submission IDs do not exactly match test transaction IDs")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA-256 without reading the whole file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes(data_dir: str | Path) -> dict[str, str]:
    """Return hashes for every file in the challenge package."""

    path = validate_data_directory(data_dir)
    return {name: sha256_file(path / name) for name in REQUIRED_FILES}
