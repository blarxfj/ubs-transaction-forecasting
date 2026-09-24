"""End-to-end feature, validation, training, and prediction workflows."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .augment import TEST_LEVEL, VALID_LEVEL, VALID_TO_TEST, add_noise, redraw_noise
from .data import data_hashes, load_transactions
from .evaluation import (
    classification_metrics,
    format_metric_line,
    metric_report,
    none_auc,
    paired_bootstrap_difference,
    recurrence_rule_predictions,
    rule_predictions,
    validate_leak_regression,
)
from .features import (
    EXPERIMENTAL_FEATURES,
    NOISE_LEVEL_FEATURES,
    SEQUENCE_FEATURES,
    TARGETED_STREAM_FEATURES,
    build_features,
)
from .model import ModelBundle, decide, fit_models, fit_predict, predict_probabilities
from .posterior import AmountPrior, learn_amount_prior
from .submission import write_prediction_artifacts
from .vocab import CLASSES


@dataclass(frozen=True)
class FeatureSpec:
    split: str
    augmentation: str | None = None
    parameters: dict[str, Any] | None = None


FEATURE_SPECS = {
    "train": FeatureSpec("train"),
    "valid": FeatureSpec("valid"),
    "test": FeatureSpec("test"),
    "trainT": FeatureSpec("train", "redraw", {**TEST_LEVEL, "seed": 0}),
    "trainV": FeatureSpec("train", "redraw", {**VALID_LEVEL, "seed": 2}),
    "validT0": FeatureSpec("valid", "add", {**VALID_TO_TEST, "seed": 0}),
    "validT1": FeatureSpec("valid", "add", {**VALID_TO_TEST, "seed": 1}),
    "validT2": FeatureSpec("valid", "add", {**VALID_TO_TEST, "seed": 2}),
}
EVALUATION_TABLES = ("train", "valid", "trainT", "trainV", "validT0", "validT1", "validT2")
TRAINING_TABLES = ("trainT", "trainV", "valid", "validT1")


def learn_prior(data_dir: str | Path) -> AmountPrior:
    """Load the unlabeled split and learn the clean amount prior."""

    return learn_amount_prior(load_transactions(data_dir, "unlabeled_pretrain"))


def _feature_job(payload: tuple[str, FeatureSpec, AmountPrior]) -> pd.DataFrame:
    data_dir, spec, amount_prior = payload
    frame = load_transactions(data_dir, spec.split)
    parameters = spec.parameters or {}
    if spec.augmentation == "redraw":
        frame = redraw_noise(frame, amount_prior, **parameters)
    elif spec.augmentation == "add":
        frame = add_noise(frame, **parameters)
    return build_features(frame, amount_prior)


def prepare_feature_tables(
    data_dir: str | Path,
    amount_prior: AmountPrior,
    *,
    names: Iterable[str] = FEATURE_SPECS,
    jobs: int = 8,
) -> dict[str, pd.DataFrame]:
    """Build requested feature tables, optionally in worker processes."""

    selected = tuple(names)
    unknown = set(selected) - set(FEATURE_SPECS)
    if unknown:
        raise ValueError(f"unknown feature tables: {sorted(unknown)}")
    payloads = [(str(data_dir), FEATURE_SPECS[name], amount_prior) for name in selected]
    if jobs == 1:
        values = [_feature_job(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(payloads))) as executor:
            values = list(executor.map(_feature_job, payloads))
    return dict(zip(selected, values, strict=True))


def _print_metric(title: str, truth: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    metrics = classification_metrics(truth, prediction)
    print(format_metric_line(title, metrics), flush=True)
    return metrics


def _cross_validated_probabilities(
    tables: dict[str, pd.DataFrame],
    train_labels: pd.Series,
    valid_labels: pd.Series,
    fold_seeds: Sequence[int],
    *,
    drop: Sequence[str] = (),
) -> dict[str, dict[int, pd.DataFrame]]:
    """Return OOF probabilities for raw valid and two independently corrupted views."""

    views = {"valid": "valid", "validT_seed0": "validT0", "validT_seed2": "validT2"}
    output = {view: {} for view in views}
    client_ids = valid_labels.index.values
    for fold_seed in fold_seeds:
        fold_outputs: dict[str, list[pd.DataFrame]] = {view: [] for view in views}
        splitter = StratifiedKFold(5, shuffle=True, random_state=fold_seed)
        for train_indices, heldout_indices in splitter.split(client_ids, valid_labels.values):
            train_clients = set(client_ids[train_indices])
            heldout_clients = set(client_ids[heldout_indices])
            training = [
                (tables["trainT"], train_labels),
                (tables["trainV"], train_labels),
                (tables["valid"][tables["valid"].client_id.isin(train_clients)], valid_labels),
                (tables["validT1"][tables["validT1"].client_id.isin(train_clients)], valid_labels),
            ]
            evaluation_tables = [
                tables[table_name][tables[table_name].client_id.isin(heldout_clients)] for table_name in views.values()
            ]
            probabilities = fit_predict(training, evaluation_tables, drop=drop)
            for view, view_probabilities in zip(views, probabilities, strict=True):
                fold_outputs[view].append(view_probabilities)
        for view in views:
            output[view][fold_seed] = pd.concat(fold_outputs[view]).reindex(valid_labels.index)
    return output


def _average_probabilities(probabilities: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not probabilities:
        raise ValueError("cannot average an empty probability collection")
    return sum(probabilities[1:], probabilities[0].copy()) / len(probabilities)


def _ablation_summary(
    candidate: dict[str, dict[int, pd.DataFrame]],
    baseline: dict[str, dict[int, pd.DataFrame]],
    labels: pd.Series,
    fold_seeds: Sequence[int],
) -> dict[str, Any]:
    rows = []
    for fold_seed in fold_seeds:
        candidate_stress = _average_probabilities(
            [candidate["validT_seed0"][fold_seed], candidate["validT_seed2"][fold_seed]]
        )
        baseline_stress = _average_probabilities(
            [baseline["validT_seed0"][fold_seed], baseline["validT_seed2"][fold_seed]]
        )
        candidate_valid_score = classification_metrics(labels, decide(candidate["valid"][fold_seed]))["macro_f1"]
        baseline_valid_score = classification_metrics(labels, decide(baseline["valid"][fold_seed]))["macro_f1"]
        candidate_stress_score = classification_metrics(labels, decide(candidate_stress))["macro_f1"]
        baseline_stress_score = classification_metrics(labels, decide(baseline_stress))["macro_f1"]
        rows.append(
            {
                "fold_seed": fold_seed,
                "valid_gain": candidate_valid_score - baseline_valid_score,
                "test_noise_gain": candidate_stress_score - baseline_stress_score,
                "candidate_valid": candidate_valid_score,
                "baseline_valid": baseline_valid_score,
                "candidate_test_noise": candidate_stress_score,
                "baseline_test_noise": baseline_stress_score,
            }
        )
    return {
        "per_fold_seed": rows,
        "retained": all(row["valid_gain"] >= 0.01 and row["test_noise_gain"] >= 0.01 for row in rows),
        "acceptance_rule": "at least +0.01 on both views for each fold seed",
    }


def evaluate_tables(
    tables: dict[str, pd.DataFrame],
    train_labels: pd.Series,
    valid_labels: pd.Series,
    *,
    fold_seeds: Sequence[int] = (0, 17),
    bootstrap_samples: int = 2000,
    run_ablations: bool = True,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run rule checks, leak regression, repeated CV, stress views, and ablations."""

    print("\n== Rule baselines (no learning) ==")
    valid_r1, valid_r4 = rule_predictions(tables["valid"], valid_labels)
    test_noise_r1, test_noise_r4 = rule_predictions(tables["validT0"], valid_labels)
    recurrence = recurrence_rule_predictions(tables["valid"], valid_labels)
    rules = {
        "valid_r1": _print_metric("valid  R1 earliest-next active stream else none", valid_labels, valid_r1),
        "valid_r4": _print_metric("valid  R4 R1 + none if >=2 refunded streams", valid_labels, valid_r4),
        "validT_r1": _print_metric("validT R1 earliest-next active stream else none", valid_labels, test_noise_r1),
        "validT_r4": _print_metric("validT R4 R1 + none if >=2 refunded streams", valid_labels, test_noise_r4),
        "recurrence_methods": {
            name: classification_metrics(valid_labels, prediction) for name, prediction in recurrence.items()
        },
    }

    print("\n== LightGBM trained on train only, evaluated on valid ==")
    valid_all, _ = fit_predict(
        [(tables["train"], train_labels)],
        [tables["valid"], tables["validT0"]],
        drop=EXPERIMENTAL_FEATURES,
    )
    valid_safe, test_noise_safe = fit_predict(
        [(tables["train"], train_labels)],
        [tables["valid"], tables["validT0"]],
        drop=(*EXPERIMENTAL_FEATURES, *NOISE_LEVEL_FEATURES),
    )
    leak = {
        "all_features": _print_metric("train->valid  all features (train leak)", valid_labels, decide(valid_all)),
        "noise_features_dropped": _print_metric(
            "train->valid  noise-level features dropped", valid_labels, decide(valid_safe)
        ),
        "noise_features_dropped_validT": _print_metric(
            "train->validT noise-level features dropped", valid_labels, decide(test_noise_safe)
        ),
    }
    validate_leak_regression(leak["all_features"]["macro_f1"], leak["noise_features_dropped"]["macro_f1"])

    if not fold_seeds:
        raise ValueError("at least one fold seed is required")
    print("\n== Repeated 5-fold CV over held-out valid clients ==")
    full = _cross_validated_probabilities(
        tables,
        train_labels,
        valid_labels,
        fold_seeds,
        drop=EXPERIMENTAL_FEATURES,
    )
    valid_probabilities = _average_probabilities(list(full["valid"].values()))
    stress_probabilities = _average_probabilities(
        [probabilities for view in ("validT_seed0", "validT_seed2") for probabilities in full[view].values()]
    )
    valid_predictions = decide(valid_probabilities)
    stress_predictions = decide(stress_probabilities)
    valid_metrics = metric_report(
        valid_labels, valid_predictions, bootstrap_samples=bootstrap_samples, bootstrap_seed=2026
    )
    stress_metrics = metric_report(
        valid_labels, stress_predictions, bootstrap_samples=bootstrap_samples, bootstrap_seed=2027
    )
    print(format_metric_line("CV -> valid  (valid noise)", valid_metrics), flush=True)
    print(format_metric_line("CV -> validT (two test-noise draws)", stress_metrics), flush=True)
    valid_auc = none_auc(valid_labels, valid_probabilities)
    stress_auc = none_auc(valid_labels, stress_probabilities)
    print(f"none-gate AUC valid={valid_auc:.4f} validT={stress_auc:.4f}", flush=True)

    fold_seed_results = {}
    for fold_seed in fold_seeds:
        seed_stress = _average_probabilities([full["validT_seed0"][fold_seed], full["validT_seed2"][fold_seed]])
        fold_seed_results[str(fold_seed)] = {
            "valid": classification_metrics(valid_labels, decide(full["valid"][fold_seed])),
            "valid_test_noise": classification_metrics(valid_labels, decide(seed_stress)),
        }

    ablations: dict[str, Any] = {}
    if run_ablations:
        for name, competing_drop in (
            ("ordered_sequence_features", TARGETED_STREAM_FEATURES),
            ("secondary_stream_timing", SEQUENCE_FEATURES),
        ):
            print(f"running candidate-feature ablation: {name}", flush=True)
            candidate = _cross_validated_probabilities(
                tables,
                train_labels,
                valid_labels,
                fold_seeds,
                drop=competing_drop,
            )
            ablations[name] = _ablation_summary(candidate, full, valid_labels, fold_seeds)

    paired_noise = paired_bootstrap_difference(
        valid_labels,
        valid_predictions,
        stress_predictions,
        samples=bootstrap_samples,
        seed=2028,
    )
    result = {
        "protocol": {
            "folds": 5,
            "fold_seeds": list(fold_seeds),
            "model_seeds": [0, 1, 2],
            "corruption_seeds": [0, 2],
            "bootstrap_samples": bootstrap_samples,
            "evidence_scope": (
                "Repeated client-held-out CV on a validation set used during development; intervals are "
                "conditional paired resampling, not an independent hidden-test guarantee."
            ),
        },
        "rules": rules,
        "leak_regression": leak,
        "fold_seed_results": fold_seed_results,
        "ablations": ablations,
        "valid": {**valid_metrics, "none_auc": valid_auc},
        "valid_test_noise": {**stress_metrics, "none_auc": stress_auc},
        "paired_test_noise_minus_valid": paired_noise,
    }
    oof = {
        **{
            f"{view}_fold{fold_seed}": probabilities
            for view, by_seed in full.items()
            for fold_seed, probabilities in by_seed.items()
        },
        "valid_combined": valid_probabilities,
        "validT_combined": stress_probabilities,
    }
    return result, oof


def train_final_model(
    tables: dict[str, pd.DataFrame],
    train_labels: pd.Series,
    valid_labels: pd.Series,
    amount_prior: AmountPrior,
) -> ModelBundle:
    """Fit the final three-seed ensemble on every labeled client."""

    training = [
        (tables["trainT"], train_labels),
        (tables["trainV"], train_labels),
        (tables["valid"], valid_labels),
        (tables["validT1"], valid_labels),
    ]
    return fit_models(training, amount_prior=amount_prior, drop=EXPERIMENTAL_FEATURES)


def predict_and_write(
    bundle: ModelBundle,
    test_features: pd.DataFrame,
    data_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict test clients and write contract-checked artifacts."""

    probabilities = predict_probabilities(bundle, test_features)
    submission = write_prediction_artifacts(
        probabilities,
        Path(data_dir) / "sample_submission.csv",
        output_dir,
    )
    print(f"submission written: {Path(output_dir) / 'submission.csv'} rows {len(submission)}")
    distribution = submission.predicted_next_recurring_merchant.value_counts().to_dict()
    print(f"predicted class distribution: {distribution}")
    return submission, probabilities


def git_revision() -> str:
    """Return the current Git revision when running inside a checkout."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_oof_artifacts(
    output_dir: str | Path,
    labels: pd.Series,
    probability_tables: dict[str, pd.DataFrame],
) -> None:
    """Retain every OOF probability view for paired reanalysis."""

    output = Path(output_dir) / "oof"
    output.mkdir(parents=True, exist_ok=True)
    for name, probabilities in sorted(probability_tables.items()):
        frame = probabilities.reindex(labels.index).copy()
        frame.insert(0, "target", labels.values)
        frame["prediction"] = decide(probabilities).reindex(labels.index).values
        frame.to_csv(output / f"{name}.csv", index_label="client_id")


def write_run_metadata(
    output_dir: str | Path,
    data_dir: str | Path,
    evaluation: dict[str, Any] | None = None,
) -> None:
    """Write stable JSON metadata and, when available, an evaluation report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "git_revision": git_revision(),
        "data_sha256": data_hashes(data_dir),
        "classes": list(CLASSES),
    }
    if evaluation is not None:
        manifest["evaluation"] = evaluation
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if evaluation is not None:
        (output / "evaluation.md").write_text(render_evaluation_markdown(evaluation), encoding="utf-8")


def render_evaluation_markdown(evaluation: dict[str, Any]) -> str:
    """Render repeated CV, paired stress uncertainty, and bounded ablations."""

    protocol = evaluation["protocol"]
    lines = [
        "# Held-out validation results",
        "",
        (
            f"Five-fold stratified CV was repeated with fold seeds "
            f"{', '.join(map(str, protocol['fold_seeds']))}. Each held-out client was excluded from "
            "both the original-valid and additionally noised training copies. Test-level noise is the "
            f"mean of corruption seeds {', '.join(map(str, protocol['corruption_seeds']))}. Intervals "
            f"are {int(100 * evaluation['valid']['bootstrap']['confidence'])}% client-bootstrap "
            f"percentile intervals ({protocol['bootstrap_samples']:,} resamples)."
        ),
        "",
        f"**Evidence scope:** {protocol['evidence_scope']}",
        "",
        "| Condition | Macro-F1 (95% CI) | Accuracy | none AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, name in (("valid", "Valid noise"), ("valid_test_noise", "Test-level noise")):
        result = evaluation[key]
        lower, upper = result["bootstrap"]["macro_f1"]
        lines.append(
            f"| {name} | {result['macro_f1']:.3f} ({lower:.3f}-{upper:.3f}) | "
            f"{result['accuracy']:.3f} | {result['none_auc']:.3f} |"
        )
    paired = evaluation["paired_test_noise_minus_valid"]
    lines.extend(
        [
            "",
            f"The paired test-noise minus valid macro-F1 difference is {paired['difference']:+.3f} "
            f"(95% CI {paired['interval'][0]:+.3f} to {paired['interval'][1]:+.3f}).",
            "",
            "## Fold-partition sensitivity",
            "",
            "| Fold seed | Valid macro-F1 | Test-level-noise macro-F1 |",
            "| ---: | ---: | ---: |",
        ]
    )
    for fold_seed, values in evaluation["fold_seed_results"].items():
        lines.append(
            f"| {fold_seed} | {values['valid']['macro_f1']:.3f} | {values['valid_test_noise']['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-class F1",
            "",
            "| Class | Valid noise (95% CI) | Test-level noise (95% CI) |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in CLASSES:
        valid = evaluation["valid"]
        test_noise = evaluation["valid_test_noise"]
        valid_bounds = valid["bootstrap"]["per_class_f1"][label]
        test_bounds = test_noise["bootstrap"]["per_class_f1"][label]
        lines.append(
            f"| {label} | {valid['per_class_f1'][label]:.3f} "
            f"({valid_bounds[0]:.3f}-{valid_bounds[1]:.3f}) | "
            f"{test_noise['per_class_f1'][label]:.3f} "
            f"({test_bounds[0]:.3f}-{test_bounds[1]:.3f}) |"
        )
    lines.extend(
        [
            "",
            "## Recurrence heuristics",
            "",
            "| Active-stream selection | Valid macro-F1 |",
            "| --- | ---: |",
        ]
    )
    for name, values in evaluation["rules"]["recurrence_methods"].items():
        lines.append(f"| {name.replace('_', ' ')} | {values['macro_f1']:.3f} |")
    lines.extend(
        [
            "",
            "## Candidate feature ablations",
            "",
            "Candidate groups are retained only for a gain of at least 0.01 on both views and both "
            "fold seeds. Unretained groups remain reproducibly computable but are dropped by the final model.",
            "",
            "| Candidate | Fold seed | Valid gain | Test-noise gain | Retained |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, ablation in evaluation["ablations"].items():
        for row in ablation["per_fold_seed"]:
            lines.append(
                f"| {name.replace('_', ' ')} | {row['fold_seed']} | {row['valid_gain']:+.3f} | "
                f"{row['test_noise_gain']:+.3f} | {'yes' if ablation['retained'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Leakage regression",
            "",
            "A train-only model using noise-quality features scores "
            f"{evaluation['leak_regression']['all_features']['macro_f1']:.3f}; dropping those features "
            "raises it to "
            f"{evaluation['leak_regression']['noise_features_dropped']['macro_f1']:.3f}. The final model "
            "uses those useful features only after every subscription-candidate row is re-noised "
            "without labels.",
            "",
        ]
    )
    return "\n".join(lines)
