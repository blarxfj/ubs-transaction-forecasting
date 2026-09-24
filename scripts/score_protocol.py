"""Score a solution's probability files under the shared protocol.

Expects ``oof_seed0.csv``, ``oof_seed1.csv``, ``oof_seed2.csv`` (development clients, with a
``fold`` column that must match the hashed folds) and ``lockbox.csv`` in one directory.
Writes ``protocol_metrics.json`` next to them and prints a one-line summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ubs_forecasting.data import load_labels
from ubs_forecasting.protocol import (
    FOLD_SEEDS,
    fold_series,
    read_probability_table,
    score_lockbox,
    score_repetitions,
    split_development,
)


def score_directory(data: str, directory: Path, *, bootstrap_samples: int = 2000) -> dict:
    train_labels = load_labels(data, "train")
    valid_labels = load_labels(data, "valid")
    development, is_valid, lockbox = split_development(train_labels, valid_labels)
    probabilities_by_seed = {}
    for seed in FOLD_SEEDS:
        path = directory / f"oof_seed{seed}.csv"
        frame = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
        if set(frame.index) != set(development.index):
            raise ValueError(f"{path} does not cover exactly the development clients")
        if "fold" in frame.columns:
            expected = fold_series(seed, list(frame.index))
            if not np.array_equal(frame["fold"].to_numpy(), expected.to_numpy()):
                raise ValueError(f"{path} fold column does not match the hashed protocol folds")
        probabilities_by_seed[seed] = read_probability_table(str(path))
    development_result = score_repetitions(
        development, probabilities_by_seed, is_valid, bootstrap_samples=bootstrap_samples
    )
    result = {"development": development_result}
    lockbox_path = directory / "lockbox.csv"
    if lockbox_path.exists():
        lockbox_probabilities = read_probability_table(str(lockbox_path))
        if set(lockbox_probabilities.index) != set(lockbox.index):
            raise ValueError(f"{lockbox_path} does not cover exactly the lockbox clients")
        result["lockbox"] = score_lockbox(lockbox, lockbox_probabilities, bootstrap_samples=bootstrap_samples)
    return result


def summary_line(name: str, result: dict) -> str:
    development = result["development"]
    parts = [
        f"{name:14s}",
        f"pooled {development['pooled_macro_f1']:.4f} "
        f"({development['pooled_ci95'][0]:.3f}-{development['pooled_ci95'][1]:.3f})",
        f"valid-only {development['valid_only_macro_f1']:.4f} "
        f"({development['valid_only_ci95'][0]:.3f}-{development['valid_only_ci95'][1]:.3f})",
    ]
    if "lockbox" in result:
        lockbox = result["lockbox"]
        parts.append(f"lockbox {lockbox['macro_f1']:.4f} ({lockbox['ci95'][0]:.3f}-{lockbox['ci95'][1]:.3f})")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--directory", required=True, action="append", help="solution directory (repeatable)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    for directory in args.directory:
        path = Path(directory)
        result = score_directory(args.data, path, bootstrap_samples=args.bootstrap)
        (path / "protocol_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(summary_line(path.name, result), flush=True)


if __name__ == "__main__":
    main()
