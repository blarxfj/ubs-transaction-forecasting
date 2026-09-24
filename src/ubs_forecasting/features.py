"""Client and per-family feature construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .posterior import AmountPrior, amount_log_probabilities
from .streams import StreamRecord, build_streams, is_active, stream_record
from .vocab import FAMILIES, FAMILY_INDEX, description_evidence

CUTOFF = pd.Timestamp("2026-01-01")
SUBSCRIPTION_KINDS = ("fam", "generic_sub", "unknown")
PRIMARY_KEYS = (
    "p_prob",
    "p_n",
    "p_last",
    "p_first",
    "p_span",
    "p_per",
    "p_gap_mad",
    "p_gap_last1",
    "p_gap_last2",
    "p_gap_last3",
    "p_gap_max",
    "p_skipped_cycle_frac",
    "p_amt",
    "p_amt_cv",
    "p_amt_trend",
    "p_amt_change_last",
    "p_nref",
    "p_ref_rate",
    "p_last_refunded",
    "p_last_ref",
    "p_odd_cur",
    "p_generic_frac",
    "p_mcc_top",
    "p_next",
    "p_overdue",
    "p_active",
)

# These values encode description/MCC noise. They are safe only after label-independent re-noising.
NOISE_LEVEL_FEATURES = ("p_prob", "s_prob", "p_generic_frac", "p_mcc_top")
SEQUENCE_FEATURES = (
    "p_gap_last1",
    "p_gap_last2",
    "p_gap_last3",
    "p_gap_max",
    "p_skipped_cycle_frac",
    "p_amt_change_last",
)
TARGETED_STREAM_FEATURES = (
    "earliest_plausible_next",
    "active_due_30_mass",
    "active_due_60_mass",
    "active_due_90_mass",
)
# Exploratory groups stay computable for reproducible ablations but are excluded from the final
# model because neither produced consistent paired gains across both fold seeds and noise views.
EXPERIMENTAL_FEATURES = (*SEQUENCE_FEATURES, *TARGETED_STREAM_FEATURES)


def row_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach grammar kind, family weights, and days relative to cutoff."""

    evidence = {description: description_evidence(description) for description in frame.description.unique()}
    output = frame.copy()
    output["kind"] = output.description.map(lambda description: evidence[description][0])
    weights = np.zeros((len(output), len(FAMILIES)))
    for row_index, description in enumerate(output.description.values):
        for family, weight in evidence[description][1].items():
            weights[row_index, FAMILY_INDEX[family]] = weight
    for family in FAMILIES:
        output[f"d_{family}"] = weights[:, FAMILY_INDEX[family]]
    output["day"] = (output.date - CUTOFF).dt.days
    return output


def client_features(
    group: pd.DataFrame,
    streams: list[StreamRecord],
) -> dict[str, int | float]:
    """Build activity and refund features shared by all family candidates."""

    type_counts = group.type.value_counts()
    features: dict[str, int | float] = {"c_rows": len(group)}
    for transaction_type in ("card_payment", "topup", "p2p_transfer", "transfer", "atm", "refund", "fee"):
        features[f"c_n_{transaction_type}"] = int(type_counts.get(transaction_type, 0))
    refunds = group[group.type == "refund"]
    subscription_refunds = refunds[refunds.kind.isin(["fam", "generic_sub"])]
    features["c_sub_refunds"] = len(subscription_refunds)
    features["c_eve_refunds"] = len(refunds) - len(subscription_refunds)
    features["c_fee_pos"] = float((group.fee > 0).mean())
    main_currency = group.currency.value_counts().index[0]
    features["c_oddcur"] = float((group.currency != main_currency).mean())
    features["c_first_day"] = int(group.day.min())
    recurring = [stream for stream in streams if stream["n"] >= 2]
    active = [stream for stream in recurring if is_active(stream)]
    ended = [stream for stream in recurring if not is_active(stream)]
    features["c_n_streams"] = len(recurring)
    features["c_n_active"] = len(active)
    features["c_n_ended"] = len(ended)
    features["c_n_active_ref"] = sum(stream["nref"] > 0 for stream in active)
    features["c_n_ended_ref"] = sum(stream["nref"] > 0 for stream in ended)
    next_days = [stream["last"] + stream["per"] for stream in active]
    features["c_min_next"] = min(next_days) if next_days else np.nan
    features["c_n_active_early"] = sum(-7 <= value <= 15 for value in next_days)
    features["c_n_recent_singles"] = sum(
        1 for stream in streams if stream["n"] == 1 and stream["last"] >= -45 and stream["post"].max() > 0.5
    )
    features["c_max_act_n"] = max((stream["n"] for stream in active), default=0)
    features["c_min_act_first"] = min((stream["first"] for stream in active), default=0)
    features["c_frac_ref_streams"] = (
        float(np.mean([stream["nref"] > 0 for stream in recurring])) if recurring else np.nan
    )
    features["c_sub_ref_last90"] = int((subscription_refunds.day >= -90).sum())
    features["c_last_sub_ref"] = float(subscription_refunds.day.max()) if len(subscription_refunds) else np.nan
    return features


def family_features(
    family: str,
    streams: list[StreamRecord],
    amount_prior: AmountPrior,
) -> dict[str, Any]:
    """Build features for one client's candidate merchant family."""

    family_index = FAMILY_INDEX[family]
    recurring = [stream for stream in streams if stream["n"] >= 2 and stream["post"][family_index] >= 0.15]
    singles = [
        stream
        for stream in streams
        if stream["n"] == 1 and stream["post"][family_index] >= 0.15 and stream["last"] >= -62
    ]
    active = [stream for stream in recurring if is_active(stream)]
    active_next = [
        (stream["last"] + (stream["per"] if np.isfinite(stream["per"]) else 30), stream) for stream in active
    ]
    features: dict[str, Any] = {
        "n_streams": float(sum(stream["post"][family_index] for stream in recurring)),
        "n_active": float(sum(stream["post"][family_index] for stream in active)),
        "n_ended_ref": float(
            sum(stream["post"][family_index] for stream in recurring if not is_active(stream) and stream["nref"] > 0)
        ),
        "earliest_plausible_next": min((value for value, _ in active_next), default=np.nan),
        "active_due_30_mass": float(sum(stream["post"][family_index] for value, stream in active_next if value <= 30)),
        "active_due_60_mass": float(sum(stream["post"][family_index] for value, stream in active_next if value <= 60)),
        "active_due_90_mass": float(sum(stream["post"][family_index] for value, stream in active_next if value <= 90)),
    }
    pool = active if active else recurring
    primary = (
        max(pool, key=lambda stream: (stream["post"][family_index] > 0.5, stream["last"], stream["n"]))
        if pool
        else None
    )
    if primary is not None:
        period = primary["per"] if np.isfinite(primary["per"]) and primary["per"] > 0 else 30.0
        values = (
            primary["post"][family_index],
            primary["n"],
            primary["last"],
            primary["first"],
            primary["span"],
            period,
            primary["gap_mad"],
            primary["gap_last1"],
            primary["gap_last2"],
            primary["gap_last3"],
            primary["gap_max"],
            primary["skipped_cycle_frac"],
            primary["amt"],
            primary["amt_cv"],
            primary["amt_trend"],
            primary["amt_change_last"],
            primary["nref"],
            primary["nref"] / primary["n"],
            primary["last_refunded"],
            primary["last_ref"],
            primary["odd_cur"],
            primary["generic_frac"],
            primary["mcc_top"],
            primary["last"] + period,
            -primary["last"] / period,
            float(is_active(primary)),
        )
        features.update(dict(zip(PRIMARY_KEYS, values, strict=True)))
    else:
        features.update({key: np.nan for key in PRIMARY_KEYS})
    single = max(singles, key=lambda stream: (stream["post"][family_index], stream["last"])) if singles else None
    features["s_prob"] = single["post"][family_index] if single else np.nan
    features["s_last"] = single["last"] if single else np.nan
    features["s_amt_logp"] = amount_log_probabilities(single["amt"], amount_prior)[family_index] if single else np.nan
    features["s_n"] = len(singles)
    return features


def build_features(frame: pd.DataFrame, amount_prior: AmountPrior) -> pd.DataFrame:
    """Build one feature row per (client, family) pair."""

    frame = row_evidence(frame)
    rows: list[dict[str, Any]] = []
    for client_id, group in frame.groupby("client_id", sort=True):
        main_currency = group.currency.value_counts().index[0]
        candidates = group[(group.type == "card_payment") & group.kind.isin(SUBSCRIPTION_KINDS)]
        refunds = group[group.type == "refund"]
        streams = (
            [stream_record(stream, refunds, main_currency, amount_prior) for stream in build_streams(candidates)]
            if len(candidates)
            else []
        )
        shared = client_features(group, streams)
        family_blocks = {family: family_features(family, streams, amount_prior) for family in FAMILIES}
        next_days = {
            family: (family_blocks[family]["p_next"] if family_blocks[family]["p_active"] == 1 else np.nan)
            for family in FAMILIES
        }
        sorted_next_days = sorted(value for value in next_days.values() if np.isfinite(value))
        for family in FAMILIES:
            block = family_blocks[family]
            own_next = next_days[family]
            other_next = [
                value for other_family, value in next_days.items() if other_family != family and np.isfinite(value)
            ]
            block["rel_next_minus_best_other"] = (
                own_next - min(other_next) if other_next and np.isfinite(own_next) else np.nan
            )
            block["n_other_active"] = len(other_next)
            block["next_rank"] = sorted_next_days.index(own_next) if np.isfinite(own_next) else np.nan
            block["e_ref"] = float(refunds[f"d_{family}"].sum())
            rows.append(
                {
                    "client_id": client_id,
                    "family": family,
                    "family_id": FAMILY_INDEX[family],
                    **block,
                    **shared,
                }
            )
    return pd.DataFrame(rows)
