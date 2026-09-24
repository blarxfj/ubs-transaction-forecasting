"""Dataset loading and reproducibility metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

TRANSACTION_SPLITS = ("unlabeled_pretrain", "train", "valid", "test")
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


def load_transactions(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load one transaction split and add normalized date fields."""

    if split not in TRANSACTION_SPLITS:
        raise ValueError(f"unknown split: {split}")
    path = Path(data_dir) / f"{split}_transactions.jsonl"
    frame = pd.read_json(path, lines=True, dtype={"mcc": str, "client_id": str})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame.timestamp.dt.tz_localize(None).dt.floor("D")
    return frame


def load_labels(data_dir: str | Path, split: str) -> pd.Series:
    """Load labels indexed by client ID."""

    if split not in {"train", "valid"}:
        raise ValueError("labels only exist for train and valid")
    path = Path(data_dir) / f"{split}_labels.csv"
    frame = pd.read_csv(path, dtype={"client_id": str})
    return frame.set_index("client_id").target_next_recurring_merchant


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
