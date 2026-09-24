"""Re-score fixed recipes and select development-only probability blends.

Run from the repository root. Private inputs remain inside the external ZIP;
only derived feature tables and predictions are cached below --output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import pickle
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ubs_forecasting.activity import activity_features, attach_activity
from ubs_forecasting.augment import TEST_LEVEL, VALID_LEVEL, VALID_TO_TEST, add_noise, redraw_noise
from ubs_forecasting.data import sha256_file, validate_transaction_frame
from ubs_forecasting.features import EXPERIMENTAL_FEATURES, build_features
from ubs_forecasting.listwise import fit_predict_listwise
from ubs_forecasting.model import DEFAULT_PARAMETERS, fit_predict
from ubs_forecasting.posterior import learn_amount_prior
from ubs_forecasting.protocol import client_folds, lockbox_ids, paired_repeated_difference, repeated_metrics
from ubs_forecasting.submission import validate_submission
from ubs_forecasting.vocab import CLASSES


def read_input(archive, name):
    with zipfile.ZipFile(archive) as source:
        members = [n for n in source.namelist() if Path(n).name == name and not n.startswith("__MACOSX")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one archive member for {name}")
        payload = io.BytesIO(source.read(members[0]))
        if name.endswith(".csv"):
            return pd.read_csv(payload, dtype={"client_id": str})
        return pd.read_json(payload, lines=True, dtype={"mcc": str, "client_id": str})


def transactions(archive, split):
    frame = read_input(archive, f"{split}_transactions.jsonl")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    validate_transaction_frame(frame, split)
    frame["date"] = frame.timestamp.dt.tz_localize(None).dt.floor("D")
    return frame


def labels(archive):
    train = read_input(archive, "train_labels.csv").set_index("client_id").target_next_recurring_merchant
    valid = read_input(archive, "valid_labels.csv").set_index("client_id").target_next_recurring_merchant
    locked = lockbox_ids(valid.index)
    dev = pd.concat([train, valid.drop(locked)])
    if not train.index.is_unique or not valid.index.is_unique or set(train.index) & set(valid.index):
        raise ValueError("label clients must be unique and split-disjoint")
    if dev.isna().any() or not dev.isin(CLASSES).all():
        raise ValueError("development labels must use the fixed eight-class vocabulary")
    return train, valid, locked, dev


def feature_job(args):
    archive, output, name = args
    path = Path(output) / "features" / f"{name}.pkl"
    if path.exists():
        return
    with (Path(output) / "prior.pkl").open("rb") as handle:
        prior = pickle.load(handle)
    split = "train" if name.startswith("train") else "valid" if name.startswith("valid") else "test"
    frame = transactions(archive, split)
    if name in ("trainT", "trainV"):
        frame = redraw_noise(
            frame, prior, **(TEST_LEVEL if name == "trainT" else VALID_LEVEL), seed=0 if name == "trainT" else 2
        )
    if name == "validT1":
        frame = add_noise(frame, **VALID_TO_TEST, seed=1)
    table = build_features(frame, prior)
    table.to_pickle(path)
    print("features", name, table.shape, flush=True)


def prepare(args):
    (args.output / "features").mkdir(parents=True, exist_ok=True)
    prior_path = args.output / "prior.pkl"
    if not prior_path.exists():
        prior = learn_amount_prior(transactions(args.data, "unlabeled_pretrain"))
        with prior_path.open("wb") as handle:
            pickle.dump(prior, handle)
    names = ("train", "trainT", "trainV", "valid", "validT1", "test")
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(feature_job, [(args.data, args.output, name) for name in names]))


def write_probability(frame, path, seed=None):
    frame = frame[list(CLASSES)].sort_index().copy()
    values = frame.to_numpy()
    if not frame.index.is_unique or frame.empty or not np.isfinite(values).all():
        raise ValueError("probabilities require unique clients and finite nonempty rows")
    if (values < 0).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("probabilities must be nonnegative and normalized")
    frame.index.name = "client_id"
    if seed is not None:
        frame.insert(0, "fold", client_folds(frame.index, seed))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, float_format="%.12g")
    temporary.replace(path)


def load_probability(path):
    return pd.read_csv(path, index_col="client_id")[list(CLASSES)]


def adjust_none(probabilities, weight):
    result = probabilities.copy()
    result["none"] *= weight
    return result.div(result.sum(axis=1), axis=0)


def trees(args, final=False):
    train, valid, locked, dev = labels(args.data)
    tables = {
        name: pd.read_pickle(args.output / "features" / f"{name}.pkl")
        for name in ("train", "trainT", "trainV", "valid", "validT1", "test")
    }
    if args.model == "activity":
        activity = {}
        for split in ("train", "valid", "test"):
            cache = args.output / "features" / f"activity_{split}.pkl"
            if not cache.exists():
                activity_features(transactions(args.data, split)).to_pickle(cache)
            activity[split] = pd.read_pickle(cache)
        for name in tables:
            split = "train" if name.startswith("train") else "valid" if name.startswith("valid") else "test"
            tables[name] = attach_activity(tables[name], activity[split])
    raw = pd.concat([tables["train"], tables["valid"]], ignore_index=True)
    destination = args.output / args.model
    destination.mkdir(exist_ok=True)
    for seed in (0,) if final else (0, 1, 2):
        path = destination / f"oof_seed{seed}.csv"
        if not final and path.exists():
            continue
        folds = client_folds(dev.index, seed)
        pieces = []
        for fold in (-1,) if final else range(5):
            ids = dev.index if final else dev.index[folds != fold]
            if final:
                evaluation = (
                    tables["test"]
                    if args.test_only
                    else pd.concat(
                        [tables["valid"][tables["valid"].client_id.isin(locked)], tables["test"]], ignore_index=True
                    )
                )
            else:
                evaluation = raw[raw.client_id.isin(dev.index[folds == fold])].reset_index(drop=True)
            training = [
                (tables[name][tables[name].client_id.isin(ids)].reset_index(drop=True), dev)
                for name in ("trainT", "trainV", "valid", "validT1")
            ]
            if any(frame.client_id.isin(evaluation.client_id).any() for frame, _ in training):
                raise ValueError("a held-out client leaked through a training view")
            if args.model == "weighted":
                training = [*training, *training[2:]]
            if args.model in ("listwise", "activity", "weighted"):
                p = fit_predict_listwise(training, [evaluation], drop=EXPERIMENTAL_FEATURES)[0]
            else:
                p = fit_predict(
                    training, [evaluation], drop=EXPERIMENTAL_FEATURES, parameters={**DEFAULT_PARAMETERS, "n_jobs": 2}
                )[0]
            pieces.append(p)
            print(args.model, seed, fold, "complete", flush=True)
        p = pd.concat(pieces)
        if final:
            if not args.test_only:
                write_probability(p.loc[locked], destination / "lockbox.csv")
            sample = read_input(args.data, "sample_submission.csv")
            write_probability(p.loc[sample.client_id], destination / "test_proba.csv")
        else:
            write_probability(p, path, seed)
            print(
                json.dumps(
                    {"model": args.model, "seed": seed, "valid": repeated_metrics(valid.drop(locked), [p], samples=200)}
                ),
                flush=True,
            )


def checkout_source(ref, destination):
    if destination.exists():
        return
    destination.mkdir(parents=True)
    import tarfile

    result = subprocess.run(["git", "archive", ref], check=True, capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        archive.extractall(destination, filter="data")


def previous1(args, final=False):
    source = args.output / "source1"
    checkout_source(args.ref1, source)
    sys.path.insert(0, str((source / "solution").resolve()))
    original = importlib.import_module("train")
    streams = importlib.import_module("streams")
    original.PARAMS["num_threads"] = 2
    train, valid, locked, dev = labels(args.data)
    destination = args.output / "iteration1"
    destination.mkdir(exist_ok=True)
    cache = destination / "features.pkl"
    if cache.exists():
        long = pd.read_pickle(cache)
    else:
        frame = pd.concat([transactions(args.data, split) for split in ("train", "valid", "test")], ignore_index=True)
        frame["ts"] = frame.timestamp.dt.tz_localize(None)
        frame["day"] = (frame.ts - pd.Timestamp("2026-01-01")).dt.total_seconds() / 86400
        frame["mcc"] = frame.mcc.astype(int)
        frame["kwfam"] = frame.description.map(streams.kw_family)
        frame["mccfam"] = frame.mcc.map(streams.MCC2FAM)
        frame["isnoise"] = frame.description.isin(streams.NOISE_DESC)
        frame["isnonrec"] = frame.description.isin(streams.NONREC)
        wide, _ = original.build_features(frame)
        long = original.to_long(wide)
        long.to_pickle(cache)
    long["y"] = (long.client_id.map(dev) == long.fam).astype(float)
    columns = [c for c in long if c not in ("client_id", "fam", "y")]
    for seed in (0,) if final else (0, 1, 2):
        path = destination / f"oof_seed{seed}.csv"
        if not final and path.exists():
            continue
        folds = client_folds(dev.index, seed)
        pieces = []
        for fold in (-1,) if final else range(5):
            ids = dev.index if final else dev.index[folds != fold]
            heldout = (
                long[~long.client_id.isin(dev.index)] if final else long[long.client_id.isin(dev.index[folds == fold])]
            )
            if final and args.test_only:
                heldout = heldout[~heldout.client_id.isin(locked)]
            training = long[long.client_id.isin(ids)].reset_index(drop=True)
            if training.client_id.isin(heldout.client_id).any():
                raise ValueError("a held-out client leaked into training")
            models, calibration = original.fit_full(training, columns, list(range(5)) if final else [seed], 450)
            p, _ = original.predict_full(models, calibration, heldout.reset_index(drop=True), columns)
            pieces.append(p)
            print("iteration1", seed, fold, "complete", flush=True)
        p = pd.concat(pieces)
        if final:
            if not args.test_only:
                write_probability(p.loc[locked], destination / "lockbox.csv")
            sample = read_input(args.data, "sample_submission.csv")
            write_probability(p.loc[sample.client_id], destination / "test_proba.csv")
        else:
            write_probability(p, path, seed)
            print(
                json.dumps(
                    {
                        "model": "iteration1",
                        "seed": seed,
                        "valid": repeated_metrics(valid.drop(locked), [p], samples=200),
                    }
                ),
                flush=True,
            )


def previous2(args):
    source = args.output / "source2"
    checkout_source(args.ref2, source)
    _, valid, locked, dev = labels(args.data)
    destination = args.output / "iteration2"
    destination.mkdir(exist_ok=True)
    for seed in (0, 1, 2):
        frame = pd.read_csv(source / f"oof_seed{seed}.csv", index_col="client_id")
        if set(frame.index) != set(dev.index) or not np.array_equal(frame.fold, client_folds(frame.index, seed)):
            raise ValueError("saved predictions do not follow the comparison protocol")
        write_probability(frame, destination / f"oof_seed{seed}.csv", seed)
    print("iteration2: exact protocol coverage and all fold assignments verified", flush=True)


def summarize(args):
    _, valid, locked, dev = labels(args.data)
    output = {}
    for model in ("main", "iteration1", "iteration2", "listwise", "activity", "weighted"):
        paths = [args.output / model / f"oof_seed{seed}.csv" for seed in (0, 1, 2)]
        if all(path.exists() for path in paths):
            probabilities = [load_probability(path) for path in paths]
            output[model] = {
                "pooled": repeated_metrics(dev, probabilities),
                "valid": repeated_metrics(valid.drop(locked), probabilities),
            }
    (args.output / "development_metrics.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


def select(args):
    if (args.output / "lockbox_opened.json").exists():
        raise RuntimeError("selection is frozen after lockbox evaluation")
    _, valid, locked, dev = labels(args.data)
    truth = valid.drop(locked)
    models = ("main", "iteration1", "iteration2", "listwise", "activity", "weighted")
    probabilities = {
        name: [load_probability(args.output / name / f"oof_seed{s}.csv") for s in (0, 1, 2)] for name in models
    }
    # Bounded grid: convex pairs and the equal-weight blend.
    candidates = [{"weights": {name: 1.0}, "none_weight": 1.0} for name in models]
    for i, first in enumerate(models):
        for second in models[i + 1 :]:
            for weight in (0.25, 0.5, 0.75):
                candidates.append({"weights": {first: weight, second: 1 - weight}, "none_weight": 1.0})
    for group in (models, ("main", "iteration1", "iteration2"), ("main", "listwise", "activity", "weighted")):
        candidates.append({"weights": dict.fromkeys(group, 1 / len(group)), "none_weight": 1.0})
    candidates = [{**recipe, "none_weight": weight} for recipe in candidates for weight in (0.8, 1.0, 1.2)]
    for recipe in candidates:
        p = [
            adjust_none(
                sum(probabilities[name][s] * weight for name, weight in recipe["weights"].items()),
                recipe["none_weight"],
            )
            for s in range(3)
        ]
        recipe["valid"] = repeated_metrics(truth, p, samples=200)
    candidates.sort(key=lambda recipe: recipe["valid"]["macro_f1"], reverse=True)
    best = candidates[0]
    (args.output / "selection.json").write_text(json.dumps(candidates, indent=2))
    (args.output / "recipe.json").write_text(json.dumps(best, indent=2))
    p = [
        adjust_none(
            sum(probabilities[name][s] * weight for name, weight in best["weights"].items()), best["none_weight"]
        )
        for s in range(3)
    ]
    destination = args.output / "selected"
    destination.mkdir(exist_ok=True)
    for seed, frame in enumerate(p):
        write_probability(frame, destination / f"oof_seed{seed}.csv", seed)
    write_probability(sum(p) / 3, destination / "oof.csv")
    metrics = {
        "pooled": repeated_metrics(dev, p),
        "valid": repeated_metrics(truth, p),
        "paired_vs_main_valid": paired_repeated_difference(truth, probabilities["main"], p),
    }
    (destination / "development_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(best, indent=2))


def final_report(args):
    marker = args.output / "final_metrics.json"
    opened = args.output / "lockbox_opened.json"
    if marker.exists() or opened.exists():
        raise RuntimeError("lockbox already opened; use recorded results")
    opened.write_text(
        json.dumps({"recipe_sha256": hashlib.sha256((args.output / "recipe.json").read_bytes()).hexdigest()})
    )
    _, valid, locked, dev = labels(args.data)
    recipe = json.loads((args.output / "recipe.json").read_text())
    source = args.output / "source2"
    for filename in ("lockbox.csv", "test_proba.csv"):
        write_probability(load_probability(source / filename), args.output / "iteration2" / filename)
        p = sum(
            load_probability(args.output / model / filename) * weight for model, weight in recipe["weights"].items()
        )
        write_probability(adjust_none(p, recipe["none_weight"]), args.output / "selected" / filename)
    sample = read_input(args.data, "sample_submission.csv")
    p = load_probability(args.output / "selected" / "test_proba.csv").reindex(sample.client_id)
    submission = pd.DataFrame(
        {"client_id": sample.client_id, "predicted_next_recurring_merchant": p.idxmax(1).to_numpy()}
    )
    validate_submission(submission, sample)
    path = args.output / "selected" / "submission.csv"
    submission.to_csv(path, index=False)
    validate_submission(pd.read_csv(path), sample)
    scores = {}
    for model in ("main", "iteration1", "iteration2", "listwise", "activity", "weighted", "selected"):
        scores[model] = repeated_metrics(valid.loc[locked], [load_probability(args.output / model / "lockbox.csv")])
    scores["submission_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_files = [Path("scripts/compare.py"), Path("uv.lock"), *sorted(Path("src/ubs_forecasting").glob("*.py"))]
    manifest = {
        "data_zip_sha256": sha256_file(args.data),
        "source_sha256": {str(file): sha256_file(file) for file in source_files},
        "previous_recipe_commits": [args.ref1, args.ref2],
        "recipe": recipe,
        "classes": list(CLASSES),
        "fold_seeds": [0, 1, 2],
        "n_folds": 5,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 2026,
        "development_clients": len(dev),
        "lockbox_clients": len(locked),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    marker.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


def verify(args):
    """Refit selected components without re-evaluating lockbox targets."""
    recipe = json.loads((args.output / "recipe.json").read_text())
    expected = (args.output / "selected" / "submission.csv").read_bytes()
    args.test_only = True
    for model in recipe["weights"]:
        if model == "iteration2":
            continue  # Fixed versioned predictions, not a newly fitted component.
        before = (args.output / model / "test_proba.csv").read_bytes()
        if model == "iteration1":
            previous1(args, final=True)
        else:
            args.model = model
            trees(args, final=True)
        after = (args.output / model / "test_proba.csv").read_bytes()
        if before != after:
            raise RuntimeError(f"non-deterministic probabilities for {model}")
    probabilities = sum(
        load_probability(args.output / model / "test_proba.csv") * weight for model, weight in recipe["weights"].items()
    )
    probabilities = adjust_none(probabilities, recipe["none_weight"])
    sample = read_input(args.data, "sample_submission.csv")
    predictions = probabilities.reindex(sample.client_id).idxmax(1)
    submission = pd.DataFrame(
        {"client_id": sample.client_id, "predicted_next_recurring_merchant": predictions.to_numpy()}
    )
    validate_submission(submission, sample)
    encoded = submission.to_csv(index=False).encode()
    if encoded != expected:
        raise RuntimeError("submission differs after refit")
    digest = hashlib.sha256(encoded).hexdigest()
    (args.output / "determinism.json").write_text(
        json.dumps({"identical": True, "submission_sha256": digest}, indent=2)
    )
    print("deterministic refit", digest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "cv",
            "predict",
            "previous1",
            "previous1-final",
            "previous2",
            "summarize",
            "select",
            "final",
            "verify",
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison"))
    parser.add_argument("--model", choices=("main", "listwise", "activity", "weighted"), default="main")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--ref1", default="70b07f0fe971d310c351953fb0b800086a6d7718")
    parser.add_argument("--ref2", default="347ccad233b0297be1d21d392a4e25f12bdd859b")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"data_zip_sha256": sha256_file(args.data), "ref1": args.ref1, "ref2": args.ref2}
    identity_path = args.output / "cache_input.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("cached inputs or baseline revisions differ; use a fresh output directory")
    identity_path.write_text(json.dumps(identity, indent=2))
    if args.stage == "prepare":
        prepare(args)
    elif args.stage in ("cv", "predict"):
        trees(args, final=args.stage == "predict")
    elif args.stage.startswith("previous1"):
        previous1(args, final=args.stage.endswith("final"))
    elif args.stage == "previous2":
        previous2(args)
    elif args.stage == "summarize":
        summarize(args)
    elif args.stage == "select":
        select(args)
    elif args.stage == "verify":
        verify(args)
    else:
        final_report(args)


if __name__ == "__main__":
    main()
