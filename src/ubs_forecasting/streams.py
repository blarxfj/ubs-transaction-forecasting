"""Currency-aware recurring stream detection and stream summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .posterior import (
    EPSILON_DESCRIPTION,
    EPSILON_REFUND,
    AmountPrior,
    amount_log_probabilities,
    description_log_likelihood,
    mcc_log_likelihood,
)
from .vocab import FAMILIES

StreamRecord = dict[str, Any]


def build_streams(group: pd.DataFrame, tolerance: float = 0.035) -> list[pd.DataFrame]:
    """Cluster log amounts by single linkage independently within each currency."""

    streams: list[pd.DataFrame] = []
    for _, currency_group in group.groupby("currency"):
        currency_group = currency_group.sort_values("amount")
        log_amounts = np.log(currency_group.amount.values)
        split_points = np.where(np.diff(log_amounts) > tolerance)[0] + 1
        for indices in np.split(np.arange(len(currency_group)), split_points):
            streams.append(currency_group.iloc[indices].sort_values("day"))
    return streams


def match_refunds(stream: pd.DataFrame, refunds: pd.DataFrame) -> pd.DataFrame:
    """Match refunds by currency, amount tolerance, and a 0-10 day delay."""

    if refunds.empty:
        return refunds
    amount = float(np.median(stream.amount))
    days = stream.day.values
    matched = refunds[(np.abs(refunds.amount / amount - 1) < 0.03) & (refunds.currency == stream.currency.iloc[0])]
    if matched.empty:
        return matched
    valid = np.array(
        [((refund_day - days >= 0) & (refund_day - days <= 10)).any() for refund_day in matched.day.values],
        dtype=bool,
    )
    return matched[valid]


def stream_record(
    stream: pd.DataFrame,
    refunds: pd.DataFrame,
    main_currency: str,
    amount_prior: AmountPrior,
) -> StreamRecord:
    """Summarize recurrence and infer a seven-family posterior for one stream."""

    days = stream.day.values
    count = len(stream)
    gaps = np.diff(days)
    period = float(np.median(gaps)) if count >= 2 else np.nan
    amount = float(np.median(stream.amount))
    matched_refunds = match_refunds(stream, refunds)
    refund_count = len(matched_refunds)
    likelihood = (
        description_log_likelihood(
            stream[[f"d_{family}" for family in FAMILIES]].values,
            EPSILON_DESCRIPTION,
        )
        + mcc_log_likelihood(stream.mcc.values)
        + amount_log_probabilities(amount, amount_prior)
    )
    if refund_count:
        likelihood += description_log_likelihood(
            matched_refunds[[f"d_{family}" for family in FAMILIES]].values,
            EPSILON_REFUND,
        )
    posterior = np.exp(likelihood - likelihood.max())
    posterior /= posterior.sum()
    last_day = int(days.max())
    recent_gaps = list(gaps[-3:][::-1])
    recent_gaps.extend([np.nan] * (3 - len(recent_gaps)))
    log_amounts = np.log(stream.amount.values)
    log_median = float(np.median(log_amounts))
    mcc_top = stream.mcc.value_counts().index[0]
    swapped = (stream.mcc.values != mcc_top).astype(float)
    generic = (stream.kind.values == "generic_sub").astype(float)
    refund_days = matched_refunds.day.values if refund_count else np.array([], dtype=float)
    return {
        "n30": int((days > -30).sum()),
        "n60": int((days > -60).sum()),
        "n90": int((days > -90).sum()),
        "n180": int((days > -180).sum()),
        "gap_last_ratio": float(gaps[-1] / period) if len(gaps) and period > 0 else np.nan,
        "n_dup": int((gaps < 5).sum()) if len(gaps) else 0,
        "n_missed": int((gaps > 1.6 * period).sum()) if len(gaps) and period > 0 else 0,
        "gap_regular": float(np.mean(np.abs(gaps / period - 1) < 0.2)) if len(gaps) and period > 0 else np.nan,
        "amt_last_dev": float(log_amounts[-1] - log_median),
        "amt_max_absdev": float(np.max(np.abs(log_amounts - log_median))),
        "n_outlier": int((np.abs(log_amounts - log_median) > 0.03).sum()),
        "swap_last1": float(swapped[-1]),
        "swap_last3": float(swapped[-3:].sum()),
        "generic_last1": float(generic[-1]),
        "generic_last3": float(generic[-3:].sum()),
        "fee_frac": float((stream.fee.values > 0).mean()),
        "dom_std": float(np.std(stream.date.dt.day.values)) if count >= 2 else np.nan,
        "ref_after_last": float((refund_days > last_day).any()) if refund_count else 0.0,
        "nref60": int((refund_days > -60).sum()) if refund_count else 0,
        "n": count,
        "first": int(days.min()),
        "last": last_day,
        "span": int(last_day - days.min()),
        "per": period,
        "gap_mad": float(np.median(np.abs(gaps - period))) if count >= 3 else np.nan,
        "gap_last1": recent_gaps[0],
        "gap_last2": recent_gaps[1],
        "gap_last3": recent_gaps[2],
        "gap_max": float(gaps.max()) if len(gaps) else np.nan,
        "skipped_cycle_frac": (float(np.mean(gaps > 1.5 * period)) if len(gaps) and period > 0 else np.nan),
        "amt": amount,
        "amt_cv": float(stream.amount.std() / stream.amount.mean()) if count >= 2 else 0.0,
        "amt_trend": float(stream.amount.iloc[-1] / stream.amount.iloc[0]) if count >= 2 else 1.0,
        "amt_change_last": (float(stream.amount.iloc[-1] / stream.amount.iloc[-2] - 1) if count >= 2 else np.nan),
        "nref": refund_count,
        "last_refunded": float(
            refund_count > 0
            and ((matched_refunds.day.values - last_day >= 0) & (matched_refunds.day.values - last_day <= 10)).any()
        ),
        "last_ref": float(matched_refunds.day.max()) if refund_count else np.nan,
        "odd_cur": float(stream.currency.iloc[0] != main_currency),
        "generic_frac": float((stream.kind == "generic_sub").mean()),
        "mcc_top": float(stream.mcc.value_counts(normalize=True).iloc[0]),
        "post": posterior,
    }


def is_active(stream: StreamRecord) -> bool:
    """Return whether a stream is still plausibly active at the cutoff."""

    period = stream["per"] if np.isfinite(stream["per"]) else 30
    return bool(stream["last"] + period >= -7)
