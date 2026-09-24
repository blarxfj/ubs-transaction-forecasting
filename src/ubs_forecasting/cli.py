"""Command-line interface for deterministic training, evaluation, and prediction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .data import load_labels, validate_data_directory
from .model import load_model, save_model
from .pipeline import (
    EVALUATION_TABLES,
    FEATURE_SPECS,
    TRAINING_TABLES,
    evaluate_tables,
    learn_prior,
    predict_and_write,
    prepare_feature_tables,
    train_final_model,
    write_run_metadata,
)


def _fold_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one fold seed is required")
    return seeds


def _common_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        default=os.environ.get("UBS_DATA_DIR"),
        help="directory containing unpacked challenge files (default: UBS_DATA_DIR)",
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel feature workers")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="ubs-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="run held-out validation")
    _common_data_arguments(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--fold-seeds", type=_fold_seeds, default=(0, 17))
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)

    train = subparsers.add_parser("train", help="fit and serialize the final ensemble")
    _common_data_arguments(train)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--output", type=Path, help="optional directory for the run manifest")

    predict = subparsers.add_parser("predict", help="predict test clients with a saved model")
    _common_data_arguments(predict)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    all_steps = subparsers.add_parser("all", help="evaluate, fit, and predict in one run")
    _common_data_arguments(all_steps)
    all_steps.add_argument("--output", type=Path, required=True)
    all_steps.add_argument("--fold-seeds", type=_fold_seeds, default=(0, 17))
    all_steps.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def _validated_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if not args.data:
        parser.error("--data is required when UBS_DATA_DIR is not set")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if hasattr(args, "bootstrap_samples") and args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return validate_data_directory(args.data)


def main(argv: list[str] | None = None) -> None:
    """Run the selected workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = _validated_arguments(args, parser)

    if args.command == "predict":
        bundle = load_model(args.model)
        if bundle.amount_prior is None:
            raise ValueError("model bundle does not contain the amount prior required for prediction")
        tables = prepare_feature_tables(data_dir, bundle.amount_prior, names=("test",), jobs=args.jobs)
        predict_and_write(bundle, tables["test"], data_dir, args.output)
        write_run_metadata(args.output, data_dir)
        return

    amount_prior = learn_prior(data_dir)
    train_labels = load_labels(data_dir, "train")
    valid_labels = load_labels(data_dir, "valid")

    if args.command == "evaluate":
        tables = prepare_feature_tables(data_dir, amount_prior, names=EVALUATION_TABLES, jobs=args.jobs)
        evaluation, _, _ = evaluate_tables(
            tables,
            train_labels,
            valid_labels,
            fold_seeds=args.fold_seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        write_run_metadata(args.output, data_dir, evaluation)
        return

    if args.command == "train":
        tables = prepare_feature_tables(data_dir, amount_prior, names=TRAINING_TABLES, jobs=args.jobs)
        bundle = train_final_model(tables, train_labels, valid_labels, amount_prior)
        save_model(bundle, args.model)
        if args.output:
            write_run_metadata(args.output, data_dir)
        print(f"model written: {args.model}")
        return

    tables = prepare_feature_tables(data_dir, amount_prior, names=FEATURE_SPECS, jobs=args.jobs)
    evaluation, _, _ = evaluate_tables(
        tables,
        train_labels,
        valid_labels,
        fold_seeds=args.fold_seeds,
        bootstrap_samples=args.bootstrap_samples,
    )
    bundle = train_final_model(tables, train_labels, valid_labels, amount_prior)
    save_model(bundle, args.output / "model.pkl")
    predict_and_write(bundle, tables["test"], data_dir, args.output)
    write_run_metadata(args.output, data_dir, evaluation)


if __name__ == "__main__":
    main()
