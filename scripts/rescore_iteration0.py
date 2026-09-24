"""Re-score the iteration-0 model (branch main) under the shared protocol.

The model, features, and training tables are exactly those of the main branch; only the client
splits change: the development set is train plus non-lockbox valid, folds are hashed per seed,
and the valid-only lockbox is predicted once by the final model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ubs_forecasting.data import load_labels
from ubs_forecasting.features import EXPERIMENTAL_FEATURES
from ubs_forecasting.model import fit_models, fit_predict, predict_probabilities
from ubs_forecasting.pipeline import learn_prior, prepare_feature_tables
from ubs_forecasting.protocol import (
    FOLD_SEEDS,
    FOLDS,
    fold_series,
    score_lockbox,
    score_repetitions,
    split_development,
    write_probability_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_labels = load_labels(args.data, "train")
    valid_labels = load_labels(args.data, "valid")
    development, is_valid, lockbox = split_development(train_labels, valid_labels)
    print(f"development {len(development)} (valid-only {int(is_valid.sum())}) lockbox {len(lockbox)}", flush=True)

    amount_prior = learn_prior(args.data)
    tables = prepare_feature_tables(
        args.data,
        amount_prior,
        names=("train", "valid", "test", "trainT", "trainV", "validT1"),
        jobs=args.jobs,
    )
    print(f"feature tables ready {time.time() - started:.0f}s", flush=True)

    def rows(name: str, clients: pd.Index) -> pd.DataFrame:
        table = tables[name]
        return table[table.client_id.isin(set(clients))]

    train_clients = train_labels.index
    valid_dev_clients = development.index[is_valid.values]
    probabilities_by_seed: dict[int, pd.DataFrame] = {}
    for seed in FOLD_SEEDS:
        folds = fold_series(seed, list(development.index))
        parts = []
        for fold in range(FOLDS):
            fit_clients = folds.index[folds.values != fold]
            heldout = folds.index[folds.values == fold]
            training = [
                (rows("trainT", fit_clients.intersection(train_clients)), train_labels),
                (rows("trainV", fit_clients.intersection(train_clients)), train_labels),
                (rows("valid", fit_clients.intersection(valid_dev_clients)), valid_labels),
                (rows("validT1", fit_clients.intersection(valid_dev_clients)), valid_labels),
            ]
            evaluation = [
                rows("train", heldout.intersection(train_clients)),
                rows("valid", heldout.intersection(valid_dev_clients)),
            ]
            parts.extend(fit_predict(training, evaluation, drop=EXPERIMENTAL_FEATURES))
            print(f"seed {seed} fold {fold} done {time.time() - started:.0f}s", flush=True)
        probabilities = pd.concat(parts).reindex(development.index)
        probabilities_by_seed[seed] = probabilities
        write_probability_table(output / f"oof_seed{seed}.csv", probabilities, folds)
    development_metrics = score_repetitions(development, probabilities_by_seed, is_valid)
    print(
        f"pooled {development_metrics['pooled_macro_f1']:.4f} valid-only {development_metrics['valid_only_macro_f1']:.4f}",
        flush=True,
    )

    bundle = fit_models(
        [
            (rows("trainT", train_clients), train_labels),
            (rows("trainV", train_clients), train_labels),
            (rows("valid", valid_dev_clients), valid_labels),
            (rows("validT1", valid_dev_clients), valid_labels),
        ],
        amount_prior=amount_prior,
        drop=EXPERIMENTAL_FEATURES,
    )
    lockbox_probabilities = predict_probabilities(bundle, rows("valid", lockbox.index)).reindex(lockbox.index)
    write_probability_table(output / "lockbox.csv", lockbox_probabilities)
    sample = pd.read_csv(Path(args.data) / "sample_submission.csv", dtype={"client_id": str})
    test_probabilities = predict_probabilities(bundle, tables["test"]).reindex(sample.client_id)
    write_probability_table(output / "test_proba.csv", test_probabilities)
    lockbox_metrics = score_lockbox(lockbox, lockbox_probabilities)
    print(f"lockbox {lockbox_metrics['macro_f1']:.4f} {time.time() - started:.0f}s", flush=True)
    (output / "metrics.json").write_text(
        json.dumps({"development": development_metrics, "lockbox": lockbox_metrics}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
