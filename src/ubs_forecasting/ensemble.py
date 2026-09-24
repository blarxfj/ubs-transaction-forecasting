"""Protocol-driven ensemble: three component models, hashed-fold CV, one-shot lockbox, blend.

Every component is fitted inside the same hashed client folds, so their out-of-fold probabilities
are paired by client and can be blended without leakage. The final models are refitted on the
whole development set once, then applied to the lockbox and test clients.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .components import keyword_streams, listwise_candidates
from .data import data_hashes, load_labels, load_transactions
from .features import EXPERIMENTAL_FEATURES
from .model import fit_listwise, predict_listwise
from .pipeline import git_revision, learn_prior, prepare_feature_tables
from .posterior import AmountPrior
from .protocol import (
    FOLD_SEEDS,
    FOLDS,
    fold_series,
    score_lockbox,
    score_repetitions,
    split_development,
    write_probability_table,
)
from .submission import write_prediction_artifacts
from .vocab import CLASSES


@dataclass(frozen=True)
class Recipe:
    """Frozen ensemble configuration. Weights were selected on development out-of-fold data only."""

    train_views: tuple[str, ...] = ("trainT", "trainV")
    valid_views: tuple[str, ...] = ("valid", "validT1")
    drop: tuple[str, ...] = EXPERIMENTAL_FEATURES
    model_seeds: tuple[int, ...] = (0, 1, 2)
    keyword_final_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    weights: dict[str, float] = field(
        default_factory=lambda: {"parser_softmax": 0.6, "keyword_streams": 0.2, "listwise_candidates": 0.2}
    )


DEFAULT_RECIPE = Recipe()


class ParserSoftmaxComponent:
    """Closed-grammar parser features, label-independent re-noising, eight-candidate softmax scorer."""

    name = "parser_softmax"

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.tables: dict[str, pd.DataFrame] = {}
        self.amount_prior: AmountPrior | None = None

    def prepare(self, data_dir: str | Path, amount_prior: AmountPrior, jobs: int) -> None:
        self.amount_prior = amount_prior
        names = ("train", "valid", "test", *self.recipe.train_views, *self.recipe.valid_views)
        self.tables = prepare_feature_tables(data_dir, amount_prior, names=dict.fromkeys(names), jobs=jobs)

    def _rows(self, name: str, clients: pd.Index) -> pd.DataFrame:
        table = self.tables[name]
        return table[table.client_id.isin(set(clients))]

    def fit_predict(
        self,
        train_clients: pd.Index,
        valid_clients: pd.Index,
        labels: pd.Series,
        predict: dict[str, pd.Index],
        seed: int | None,
    ) -> dict[str, pd.DataFrame]:
        training = [(self._rows(view, train_clients), labels) for view in self.recipe.train_views]
        training += [(self._rows(view, valid_clients), labels) for view in self.recipe.valid_views]
        bundle = fit_listwise(
            training, amount_prior=self.amount_prior, drop=self.recipe.drop, seeds=self.recipe.model_seeds
        )
        return {
            split: predict_listwise(bundle, self._rows(split, clients)).reindex(clients)
            for split, clients in predict.items()
        }


class KeywordStreamsComponent:
    """Keyword family assignment, shared binary scorer, logistic second stage."""

    name = "keyword_streams"

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.long: pd.DataFrame | None = None

    def prepare(self, data_dir: str | Path, amount_prior: AmountPrior, jobs: int) -> None:
        frames = pd.concat(
            [load_transactions(data_dir, split) for split in ("train", "valid", "test")], ignore_index=True
        )
        self.long = keyword_streams.to_long(keyword_streams.build_wide_features(frames))

    def _rows(self, clients: pd.Index) -> pd.DataFrame:
        assert self.long is not None
        return self.long[self.long.client_id.isin(set(clients))].reset_index(drop=True)

    def fit_predict(
        self,
        train_clients: pd.Index,
        valid_clients: pd.Index,
        labels: pd.Series,
        predict: dict[str, pd.Index],
        seed: int | None,
    ) -> dict[str, pd.DataFrame]:
        seeds = (seed,) if seed is not None else self.recipe.keyword_final_seeds
        model = keyword_streams.fit(self._rows(train_clients.union(valid_clients)), labels, seeds=seeds)
        return {
            split: keyword_streams.predict(model, self._rows(clients)).reindex(clients)
            for split, clients in predict.items()
        }


class ListwiseCandidatesComponent:
    """Amount-kernel family reconstruction, eight candidate rows, listwise symmetric-tree ranker."""

    name = "listwise_candidates"

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.tensor: listwise_candidates.CandidateTensor | None = None

    def prepare(self, data_dir: str | Path, amount_prior: AmountPrior, jobs: int) -> None:
        frames = pd.concat(
            [load_transactions(data_dir, split) for split in ("train", "valid", "test")], ignore_index=True
        )
        self.tensor = listwise_candidates.build_tensor(frames)

    def _matrix(self, clients: pd.Index) -> np.ndarray:
        assert self.tensor is not None
        return listwise_candidates.candidate_matrix(self.tensor.subset(list(clients)))

    def fit_predict(
        self,
        train_clients: pd.Index,
        valid_clients: pd.Index,
        labels: pd.Series,
        predict: dict[str, pd.Index],
        seed: int | None,
    ) -> dict[str, pd.DataFrame]:
        fit_clients = train_clients.union(valid_clients)
        targets = labels.reindex(fit_clients).map(list(CLASSES).index).to_numpy()
        model = listwise_candidates.fit(self._matrix(fit_clients), targets, seed=0 if seed is None else seed)
        return {
            split: listwise_candidates.predict(model, self._matrix(clients), clients).reindex(clients)
            for split, clients in predict.items()
        }


def build_components(recipe: Recipe, names: Sequence[str] | None = None) -> list[Any]:
    available = {
        ParserSoftmaxComponent.name: ParserSoftmaxComponent,
        KeywordStreamsComponent.name: KeywordStreamsComponent,
        ListwiseCandidatesComponent.name: ListwiseCandidatesComponent,
    }
    selected = list(available) if names is None else list(names)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")
    return [available[name](recipe) for name in selected]


def blend(probabilities: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted arithmetic mean of component probabilities over the components present."""

    present = {name: weight for name, weight in weights.items() if name in probabilities and weight > 0}
    if not present:
        raise ValueError("no weighted component probabilities to blend")
    total = sum(present.values())
    index = next(iter(probabilities.values())).index
    output = sum(
        probabilities[name].reindex(index)[list(CLASSES)] * (weight / total) for name, weight in present.items()
    )
    assert isinstance(output, pd.DataFrame)
    return output


def run_ensemble(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    recipe: Recipe = DEFAULT_RECIPE,
    component_names: Sequence[str] | None = None,
    fold_seeds: Sequence[int] = FOLD_SEEDS,
    jobs: int = 8,
    skip_cv: bool = False,
) -> dict[str, Any]:
    """Run protocol CV for every component, refit on development, predict lockbox and test, blend."""

    started = time.time()
    output = Path(output_dir)
    (output / "components").mkdir(parents=True, exist_ok=True)
    train_labels = load_labels(data_dir, "train")
    valid_labels = load_labels(data_dir, "valid")
    development, is_valid, lockbox = split_development(train_labels, valid_labels)
    labels = pd.concat([train_labels, valid_labels])
    train_clients = train_labels.index
    valid_dev_clients = development.index[is_valid.values]
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv", dtype={"client_id": str})
    test_clients = pd.Index(sample.client_id)
    print(
        f"development {len(development)} (valid-only {len(valid_dev_clients)}) "
        f"lockbox {len(lockbox)} test {len(test_clients)}",
        flush=True,
    )

    amount_prior = learn_prior(data_dir)
    components = build_components(recipe, component_names)
    for component in components:
        component.prepare(data_dir, amount_prior, jobs)
        print(f"{component.name}: features ready {time.time() - started:.0f}s", flush=True)

    metrics: dict[str, Any] = {"components": {}, "recipe": recipe_dict(recipe)}
    oof: dict[str, dict[int, pd.DataFrame]] = {}
    if not skip_cv:
        for component in components:
            oof[component.name] = {}
            for seed in fold_seeds:
                folds = fold_series(seed, list(development.index))
                parts = []
                for fold in range(FOLDS):
                    fit_clients = folds.index[folds.values != fold]
                    heldout = folds.index[folds.values == fold]
                    predicted = component.fit_predict(
                        fit_clients.intersection(train_clients),
                        fit_clients.intersection(valid_dev_clients),
                        labels,
                        {
                            "train": heldout.intersection(train_clients),
                            "valid": heldout.intersection(valid_dev_clients),
                        },
                        seed,
                    )
                    parts.extend(predicted.values())
                    print(f"{component.name}: seed {seed} fold {fold} done {time.time() - started:.0f}s", flush=True)
                probabilities = pd.concat(parts).reindex(development.index)
                oof[component.name][seed] = probabilities
                write_probability_table(
                    output / "components" / f"{component.name}_oof_seed{seed}.csv", probabilities, folds
                )
            metrics["components"][component.name] = {
                "development": score_repetitions(development, oof[component.name], is_valid)
            }
            print(summary_line(component.name, metrics["components"][component.name]), flush=True)
        blended = {
            seed: blend({name: by_seed[seed] for name, by_seed in oof.items()}, recipe.weights) for seed in fold_seeds
        }
        for seed, probabilities in blended.items():
            write_probability_table(
                output / f"oof_seed{seed}.csv", probabilities, fold_series(seed, list(development.index))
            )
        write_probability_table(
            output / "oof.csv", blended[fold_seeds[0]], fold_series(fold_seeds[0], list(development.index))
        )
        metrics["ensemble"] = {"development": score_repetitions(development, blended, is_valid)}

    final: dict[str, dict[str, pd.DataFrame]] = {}
    for component in components:
        final[component.name] = component.fit_predict(
            train_clients, valid_dev_clients, labels, {"valid": lockbox.index, "test": test_clients}, None
        )
        write_probability_table(output / "components" / f"{component.name}_lockbox.csv", final[component.name]["valid"])
        write_probability_table(
            output / "components" / f"{component.name}_test_proba.csv", final[component.name]["test"]
        )
        metrics["components"].setdefault(component.name, {})["lockbox"] = score_lockbox(
            lockbox, final[component.name]["valid"]
        )
        print(f"{component.name}: final fit done {time.time() - started:.0f}s", flush=True)
    lockbox_probabilities = blend({name: value["valid"] for name, value in final.items()}, recipe.weights)
    test_probabilities = blend({name: value["test"] for name, value in final.items()}, recipe.weights)
    write_probability_table(output / "lockbox.csv", lockbox_probabilities)
    write_probability_table(output / "test_proba.csv", test_probabilities)
    submission = write_prediction_artifacts(test_probabilities, Path(data_dir) / "sample_submission.csv", output)
    metrics.setdefault("ensemble", {})["lockbox"] = score_lockbox(lockbox, lockbox_probabilities)
    metrics["ensemble"]["submission_distribution"] = (
        submission.predicted_next_recurring_merchant.value_counts().sort_index().to_dict()
    )
    metrics["git_revision"] = git_revision()
    metrics["data_sha256"] = data_hashes(data_dir)
    metrics["seconds"] = round(time.time() - started)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_line("ensemble", metrics["ensemble"]), flush=True)
    return metrics


def recipe_dict(recipe: Recipe) -> dict[str, Any]:
    return {
        "train_views": list(recipe.train_views),
        "valid_views": list(recipe.valid_views),
        "drop": list(recipe.drop),
        "model_seeds": list(recipe.model_seeds),
        "keyword_final_seeds": list(recipe.keyword_final_seeds),
        "weights": dict(recipe.weights),
    }


def summary_line(name: str, result: dict[str, Any]) -> str:
    parts = [f"{name:20s}"]
    if "development" in result:
        development = result["development"]
        parts.append(
            f"pooled {development['pooled_macro_f1']:.4f} "
            f"({development['pooled_ci95'][0]:.3f}-{development['pooled_ci95'][1]:.3f})"
        )
        parts.append(
            f"valid-only {development['valid_only_macro_f1']:.4f} "
            f"({development['valid_only_ci95'][0]:.3f}-{development['valid_only_ci95'][1]:.3f})"
        )
    if "lockbox" in result:
        lockbox = result["lockbox"]
        parts.append(f"lockbox {lockbox['macro_f1']:.4f} ({lockbox['ci95'][0]:.3f}-{lockbox['ci95'][1]:.3f})")
    return " | ".join(parts)
