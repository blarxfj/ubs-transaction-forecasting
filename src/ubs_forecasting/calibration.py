"""Parser-conditioned diagnostics for corruption calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .augment import TEST_LEVEL, VALID_LEVEL, VALID_TO_TEST, add_noise, redraw_noise
from .data import load_labels, load_transactions
from .features import row_evidence
from .posterior import AmountPrior
from .streams import build_streams
from .vocab import (
    EVERYDAY_MCC,
    FAMILY_MCC,
    description_evidence,
    parse_description,
)

UNAMBIGUOUS_GENERIC_SUBSCRIPTION = frozenset({"subscription charge", "member plan", "digital service"})


def noise_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Measure noise on regular detected streams using the final production parser.

    These are parser/detector-conditioned estimates, not exact noise rates in the data. Sparse,
    bimonthly, heavily masked, and ambiguous streams are underrepresented.
    """

    enriched = row_evidence(frame)
    card_payments = enriched[enriched.type == "card_payment"]
    everyday = card_payments[card_payments.kind == "everyday"]
    generic_everyday_count = int((card_payments.kind == "generic_eve").sum())
    everyday_denominator = generic_everyday_count + len(everyday)
    everyday_mask = generic_everyday_count / max(everyday_denominator, 1)
    everyday_expected_mcc = everyday.description.map(
        lambda description: EVERYDAY_MCC[next(iter(parse_description(description)[0]))]
    )
    everyday_mcc_mismatch = float((everyday.mcc != everyday_expected_mcc).mean())

    candidates = card_payments[card_payments.kind.isin(["fam", "generic_sub"])]
    row_count = generic_count = family_mcc_count = family_name_count = transformed_count = 0
    detected_streams = 0
    for _, client in candidates.groupby("client_id"):
        for stream in build_streams(client):
            if len(stream) < 4:
                continue
            gaps = np.diff(stream.day.values)
            if not 10 <= np.median(gaps) <= 40:
                continue
            family_weights: dict[str, float] = {}
            for description in stream.description[stream.kind == "fam"]:
                for family, weight in description_evidence(description)[1].items():
                    family_weights[family] = family_weights.get(family, 0) + weight
            if not family_weights:
                continue
            family = max(sorted(family_weights), key=lambda value: family_weights[value])
            if family_weights[family] < 2:
                continue
            detected_streams += 1
            row_count += len(stream)
            generic_count += int(stream.description.isin(UNAMBIGUOUS_GENERIC_SUBSCRIPTION).sum())
            family_mcc_count += int((stream.mcc == FAMILY_MCC[family]).sum())
            family_rows = stream[stream.kind == "fam"]
            family_name_count += len(family_rows)
            transformed_count += sum(
                parse_description(description)[1] > 0 or not parse_description(description)[2]
                for description in family_rows.description
            )
    return {
        "everyday_mask": float(everyday_mask),
        "everyday_mcc_mismatch": everyday_mcc_mismatch,
        "subscription_mask_estimate": float(4 / 3 * generic_count / max(row_count, 1)),
        "subscription_family_mcc_match": float(family_mcc_count / max(row_count, 1)),
        "subscription_affixed_or_truncated": float(transformed_count / max(family_name_count, 1)),
        "detected_streams": detected_streams,
        "detected_stream_rows": row_count,
    }


def candidate_generic_rate_by_label(frame: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    """Audit generic-description rates over every subscription-candidate card row."""

    enriched = row_evidence(frame)
    candidates = enriched[
        (enriched.type == "card_payment") & enriched.kind.isin(["fam", "generic_sub", "unknown"])
    ].copy()
    candidates["is_none"] = candidates.client_id.map(labels).eq("none")
    candidates["generic"] = candidates.description.isin(UNAMBIGUOUS_GENERIC_SUBSCRIPTION)
    output = {}
    for name, is_none in (("none", True), ("non_none", False)):
        subset = candidates[candidates.is_none == is_none]
        output[name] = {
            "rows": len(subset),
            "generic_rows": int(subset.generic.sum()),
            "generic_rate": float(subset.generic.mean()),
        }
    return output


def calibration_report(data_dir: str | Path, amount_prior: AmountPrior) -> dict[str, Any]:
    """Measure raw and augmented noise, including the candidate-row shortcut audit."""

    raw = {split: load_transactions(data_dir, split) for split in ("unlabeled_pretrain", "train", "valid", "test")}
    train_test = redraw_noise(raw["train"], amount_prior, **TEST_LEVEL, seed=0)
    train_valid = redraw_noise(raw["train"], amount_prior, **VALID_LEVEL, seed=2)
    valid_test_0 = add_noise(raw["valid"], **VALID_TO_TEST, seed=0)
    valid_test_2 = add_noise(raw["valid"], **VALID_TO_TEST, seed=2)
    frames = {
        **raw,
        "train_redrawn_test": train_test,
        "train_redrawn_valid": train_valid,
        "valid_test_seed0": valid_test_0,
        "valid_test_seed2": valid_test_2,
    }
    train_labels = load_labels(data_dir, "train")
    return {
        "detector_limitations": (
            "Statistics are conditioned on the final parser and regular >=4-payment detected streams; "
            "they are not exact noise rates in the data. The mask estimate uses a 4/3 "
            "adjustment because monthly plan is both a mobile name and a generic mask."
        ),
        "noise_statistics": {name: noise_statistics(frame) for name, frame in frames.items()},
        "train_candidate_generic_rate_before": candidate_generic_rate_by_label(raw["train"], train_labels),
        "train_candidate_generic_rate_after": candidate_generic_rate_by_label(train_test, train_labels),
    }


def write_calibration_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write machine-readable and Markdown calibration artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Noise calibration",
        "",
        report["detector_limitations"],
        "",
        "| Split/view | Everyday mask | Everyday MCC mismatch | Subscription mask estimate | "
        "Family-MCC match | Affixed/truncated | Streams |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["noise_statistics"].items():
        lines.append(
            f"| {name} | {values['everyday_mask']:.3f} | "
            f"{values['everyday_mcc_mismatch']:.3f} | "
            f"{values['subscription_mask_estimate']:.3f} | "
            f"{values['subscription_family_mcc_match']:.3f} | "
            f"{values['subscription_affixed_or_truncated']:.3f} | "
            f"{values['detected_streams']} |"
        )
    lines.extend(
        [
            "",
            "## All train subscription candidates",
            "",
            "| State | none generic rate | non-none generic rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for state, key in (
        ("Before redraw", "train_candidate_generic_rate_before"),
        ("After redraw", "train_candidate_generic_rate_after"),
    ):
        values = report[key]
        lines.append(f"| {state} | {values['none']['generic_rate']:.3f} | {values['non_none']['generic_rate']:.3f} |")
    (output / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
