"""Reproduce the development experiments behind CHANGES.md under the shared protocol.

Every variant fits the eight-candidate softmax scorer inside the hashed client folds of seeds
0, 1, 2 and scores validation-only and pooled macro-F1 on the out-of-fold probabilities. Feature
tables are cached under ``--cache`` so the variants only refit the scorer. The paired client
bootstrap is against a reference solution's ``oof_seed{0,1,2}.csv`` (default: the tracked
iteration-3 deliverables). The lockbox is never touched here.

    uv run python scripts/protocol_experiments.py --data "$UBS_DATA_DIR" --cache artifacts/cache \\
        --output artifacts/experiments --variant base --variant pseudo90_w03
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ubs_forecasting.data import load_labels
from ubs_forecasting.evaluation import paired_bootstrap_difference
from ubs_forecasting.features import EXPERIMENTAL_FEATURES
from ubs_forecasting.model import DEFAULT_PARAMETERS, fit_listwise, predict_listwise
from ubs_forecasting.pipeline import FEATURE_SPECS, learn_prior, prepare_feature_tables, pseudo_label_tables
from ubs_forecasting.protocol import (
    FOLD_SEEDS,
    FOLDS,
    argmax_labels,
    fold_series,
    score_repetitions,
    split_development,
    write_probability_table,
)
from ubs_forecasting.vocab import CLASSES

TRAIN_VIEWS = ("trainT", "trainV")
VALID_VIEWS = ("valid", "validT1")


def pseudo(*tables: str, weight: float = 1.0, family_only: bool = True) -> list[dict[str, Any]]:
    return [{"table": table, "weight": weight, "family_only": family_only} for table in tables]


VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    "pseudo90_w1": {"pseudo": pseudo("unlabeled90T", "unlabeled90V")},
    "pseudo90_w03": {"pseudo": pseudo("unlabeled90T", "unlabeled90V", weight=0.3)},
    "pseudo90_full_w03": {"pseudo": pseudo("unlabeled90T", "unlabeled90V", weight=0.3, family_only=False)},
    "pseudo90_180_w1": {"pseudo": pseudo("unlabeled90T", "unlabeled90V", "unlabeled180T", "unlabeled180V")},
    "pseudo90_train_w1": {"pseudo": pseudo("unlabeled90T", "unlabeled90V", "train90T", "train90V")},
    "pseudo90_clean_w1": {"pseudo": pseudo("unlabeled90")},
    "seeds5": {"model_seeds": [0, 1, 2, 3, 4]},
    "regularized": {"params": {"min_child_samples": 40, "reg_lambda": 3.0}},
    "earliest_summaries": {"earliest_summaries": True},
    "drop_unsupported": {"drop_unsupported": True},
    "earliest_summaries_drop_unsupported": {"earliest_summaries": True, "drop_unsupported": True},
}


def cached_tables(data: Path, cache: Path, names: list[str], jobs: int) -> tuple[dict, dict]:
    """Feature tables and pseudo labels, built once and pickled under ``cache``."""

    cache.mkdir(parents=True, exist_ok=True)
    prior_path = cache / "prior.pkl"
    if prior_path.exists():
        prior = pickle.loads(prior_path.read_bytes())
    else:
        prior = learn_prior(data)
        prior_path.write_bytes(pickle.dumps(prior))
    missing = [name for name in names if not (cache / f"{name}.pkl").exists()]
    if missing:
        for name, table in prepare_feature_tables(data, prior, names=missing, jobs=jobs).items():
            (cache / f"{name}.pkl").write_bytes(pickle.dumps(table))
    pseudo_names = [name for name in names if FEATURE_SPECS[name].shift is not None]
    missing_labels = [name for name in pseudo_names if not (cache / f"{name}_labels.pkl").exists()]
    if missing_labels:
        for name, table in pseudo_label_tables(data, prior, missing_labels, jobs=jobs).items():
            (cache / f"{name}_labels.pkl").write_bytes(pickle.dumps(table))
    tables = {name: pickle.loads((cache / f"{name}.pkl").read_bytes()) for name in names}
    labels = {name: pickle.loads((cache / f"{name}_labels.pkl").read_bytes()) for name in pseudo_names}
    return tables, labels


def rows(table: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    return table[table.client_id.isin(set(clients))]


def drop_unsupported_clients(table: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Drop training clients whose label family has neither a recurring stream nor a single event."""

    label_rows = table[table.family == table.client_id.map(labels)]
    unsupported = set(label_rows[label_rows.p_prob.isna() & (label_rows.s_n == 0)].client_id)
    return table[~table.client_id.isin(unsupported)]


def run_variant(name: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output) / name
    output.mkdir(parents=True, exist_ok=True)
    pseudo_views = config.get("pseudo", [])
    names = [*TRAIN_VIEWS, *VALID_VIEWS, "train", "valid", *(view["table"] for view in pseudo_views)]
    tables, pseudo_labels = cached_tables(Path(args.data), Path(args.cache), list(dict.fromkeys(names)), args.jobs)
    params = {**DEFAULT_PARAMETERS, **config.get("params", {})}
    model_seeds = tuple(config.get("model_seeds", (0, 1, 2)))

    train_labels = load_labels(args.data, "train")
    valid_labels = load_labels(args.data, "valid")
    development, is_valid, lockbox = split_development(train_labels, valid_labels)
    labels = pd.concat([train_labels, valid_labels])
    train_clients = train_labels.index
    valid_dev = development.index[is_valid.values]

    oof: dict[int, pd.DataFrame] = {}
    for seed in FOLD_SEEDS:
        folds = fold_series(seed, list(development.index))
        parts = []
        for fold in range(FOLDS):
            fit_clients = folds.index[folds.values != fold]
            heldout = folds.index[folds.values == fold]
            training = [(rows(tables[view], fit_clients.intersection(train_clients)), labels) for view in TRAIN_VIEWS]
            training += [(rows(tables[view], fit_clients.intersection(valid_dev)), labels) for view in VALID_VIEWS]
            if config.get("drop_unsupported"):
                training = [(drop_unsupported_clients(table, labels), labels) for table, _ in training]
            weights = [1.0] * len(training)
            family_only = [False] * len(training)
            for view in pseudo_views:
                table, soft = tables[view["table"]], pseudo_labels[view["table"]]
                if FEATURE_SPECS[view["table"]].split in ("train", "valid"):
                    table = rows(table, fit_clients)
                if view["family_only"]:
                    table = rows(table, soft.index[soft[list(CLASSES[:-1])].sum(axis=1) > 0])
                training.append((table, soft))
                weights.append(view["weight"])
                family_only.append(view["family_only"])
            bundle = fit_listwise(
                training,
                drop=EXPERIMENTAL_FEATURES,
                seeds=model_seeds,
                parameters=params,
                weights=weights,
                family_only=family_only,
                earliest_summaries=bool(config.get("earliest_summaries", False)),
            )
            for split, table_name in (("train", "train"), ("valid", "valid")):
                clients = heldout.intersection(train_clients if split == "train" else valid_dev)
                parts.append(predict_listwise(bundle, rows(tables[table_name], clients)).reindex(clients))
            print(f"{name}: seed {seed} fold {fold} done {time.time() - started:.0f}s", flush=True)
        oof[seed] = pd.concat(parts).reindex(development.index)
        write_probability_table(output / f"oof_seed{seed}.csv", oof[seed], folds)

    result = score_repetitions(development, oof, is_valid, bootstrap_samples=args.bootstrap)
    paired = []
    for seed in FOLD_SEEDS:
        reference = pd.read_csv(Path(args.reference) / f"oof_seed{seed}.csv", dtype={"client_id": str})
        reference = reference.set_index("client_id").reindex(valid_dev)
        paired.append(
            paired_bootstrap_difference(
                development.reindex(valid_dev),
                argmax_labels(reference),
                argmax_labels(oof[seed].reindex(valid_dev)),
                samples=args.bootstrap,
            )
        )
    summary = {
        "variant": name,
        "config": config,
        "valid_only_macro_f1": result["valid_only_macro_f1"],
        "valid_only_ci95": result["valid_only_ci95"],
        "valid_only_per_seed": [result["seeds"][str(seed)]["valid_only_macro_f1"] for seed in FOLD_SEEDS],
        "pooled_macro_f1": result["pooled_macro_f1"],
        "valid_only_per_class_f1": result["valid_only_per_class_f1"],
        "paired_minus_reference": {
            "mean": float(np.mean([item["difference"] for item in paired])),
            "per_seed": [item["difference"] for item in paired],
            "interval_per_seed": [item["interval"] for item in paired],
        },
        "seconds": round(time.time() - started),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{name:36s} valid-only {summary['valid_only_macro_f1']:.4f} "
        f"({summary['valid_only_ci95'][0]:.3f}-{summary['valid_only_ci95'][1]:.3f}) "
        f"seeds {' '.join(f'{v:.3f}' for v in summary['valid_only_per_seed'])} | "
        f"pooled {summary['pooled_macro_f1']:.4f} | "
        f"paired vs reference {summary['paired_minus_reference']['mean']:+.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True, help="directory for pickled feature tables")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="deliverables", help="directory with reference oof_seed*.csv")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="repeatable; default all")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    for name in args.variant or list(VARIANTS):
        run_variant(name, VARIANTS[name], args)


if __name__ == "__main__":
    main()
