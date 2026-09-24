"""Keyword-assigned stream component: per-(client, family) binary scorer with a logistic second stage.

Families are assigned to card payments through a keyword table, then generic or truncated names are
linked to the nearest named amount anchor of the same client. About seventy per-family stream
statistics are unpivoted into one row per (client, family) with family-agnostic column names so a
single binary model shares churn and timing signal across families. A multinomial logistic
regression fitted on inner out-of-fold scores turns the seven family scores into eight-class
probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..protocol import fold_of
from ..vocab import CLASSES, FAMILIES

CUTOFF = pd.Timestamp("2026-01-01")
MCC_FAMILY = {4814: "mobile", 5732: "cloud", 5734: "software", 6300: "insurance", 7997: "gym"}
FAMILY_MAIN_MCC = {
    "cloud": 5732,
    "gym": 7997,
    "insurance": 6300,
    "mobile": 4814,
    "music": 5812,
    "software": 5734,
    "streaming": 5812,
}
KEYWORDS = {
    "cloud": ["cloud", "backup", "storage", "service plan"],
    "gym": ["gym", "fit", "fitness", "club", "urban"],
    "insurance": ["cover", "insurance", "policy", "safe"],
    "mobile": ["phone", "contract", "service bill", "bill"],
    "music": ["audio", "member pass", "pass"],
    "software": ["saas", "productivity", "suite", "software", "prod"],
    "streaming": ["media", "video", "stream", "streaming"],
}
NOISE_DESCRIPTIONS = frozenset({"merchant charge", "service payment", "card purchase", "digital order"})
NON_RECURRING = frozenset(
    {
        "coffee shop",
        "casual dining",
        "electronics shop",
        "online marketplace",
        "fresh foods",
        "grocery store",
        "neighborhood market",
        "pharmacy",
        "hotel booking",
        "ride share",
        "atm withdrawal",
        "salary",
        "p2p send",
        "p2p receive",
        "service fee",
    }
)
GENERIC = frozenset({"monthly plan", "member plan", "digital service", "subscription charge"})
CLEAN = (
    frozenset(
        {
            "cloud access",
            "cloud backup",
            "service plan",
            "storage plan",
            "urban gym",
            "gym membership",
            "fit club",
            "fitness monthly",
            "cover plan",
            "safe cover",
            "policy premium",
            "insurance monthly",
            "monthly plan",
            "phone contract",
            "service bill",
            "digital plus",
            "member pass",
            "audio streaming",
            "premium plan",
            "saas billing",
            "productivity suite",
            "software access",
            "media streaming",
            "video access",
        }
    )
    | GENERIC
)
CURRENCIES = ("chf", "eur", "usd", "gbp")
PARAMETERS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
}
ROUNDS = 450
INNER_FOLD_OFFSET = 1000


def keyword_family(description: str) -> str | None:
    tokens = description.split()
    for family, keywords in KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword:
                if keyword in description:
                    return family
            elif keyword in tokens:
                return family
    return None


def prepare_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fractional days to cutoff, keyword and MCC families, and noise flags."""

    output = frame.copy()
    output["ts"] = output.timestamp.dt.tz_localize(None)
    output["day"] = (output.ts - CUTOFF).dt.total_seconds() / 86400.0
    output["mcc"] = output.mcc.astype(int)
    output["kwfam"] = output.description.map(keyword_family)
    output["mccfam"] = output.mcc.map(MCC_FAMILY)
    output["isnoise"] = output.description.isin(NOISE_DESCRIPTIONS)
    output["isnonrec"] = output.description.isin(NON_RECURRING)
    return output


def assign_family(group: pd.DataFrame, link_tolerance: float = 0.08) -> pd.Series:
    """Family per candidate event: keyword first, then amount anchors, then MCC."""

    family = group.kwfam.copy()
    named = group[group.kwfam.notna()]
    if len(named):
        anchors: list[tuple[str, float, int]] = []
        for anchor_family, anchor_group in named.groupby("kwfam"):
            amounts = np.sort(np.log(anchor_group.amount.values))
            clusters = np.concatenate([[0], np.cumsum(np.diff(amounts) > 0.04)])
            for cluster in np.unique(clusters):
                members = clusters == cluster
                anchors.append((anchor_family, float(np.median(amounts[members])), int(members.sum())))
        anchor_amounts = np.array([anchor[1] for anchor in anchors])
        anchor_support = np.array([anchor[2] for anchor in anchors])
        for position in np.where(family.isna())[0]:
            distance = np.abs(anchor_amounts - np.log(group.amount.iloc[position]))
            within = distance < link_tolerance
            if within.any():
                best = int(np.argmin(distance / np.sqrt(anchor_support)))
                if distance[best] >= link_tolerance:
                    best = int(np.argmin(np.where(within, distance, 9)))
                family.iloc[position] = anchors[best][0]
    remaining = family.isna()
    family[remaining] = group.mccfam[remaining]
    remaining = family.isna()
    family[remaining & (group.mcc == 5812)] = "amb5812"
    return family.fillna("unknown")


def timeline_features(group: pd.DataFrame, prefix: str, refunds: pd.DataFrame | None) -> dict[str, float]:
    """Stream statistics for the events of one family of one client."""

    features: dict[str, float] = {}
    days = group.day.values
    amounts = group.amount.values
    count = len(days)
    features[f"{prefix}_n"] = count
    if count == 0:
        return features
    gaps = np.diff(days)
    features[f"{prefix}_first"] = days[0]
    features[f"{prefix}_last"] = days[-1]
    features[f"{prefix}_span"] = days[-1] - days[0]
    features[f"{prefix}_n_named"] = group.kwfam.notna().sum()
    features[f"{prefix}_frac_generic"] = group.kwfam.isna().mean()
    features[f"{prefix}_generic_last3"] = group.kwfam.tail(3).isna().sum()
    features[f"{prefix}_generic_last1"] = float(pd.isna(group.kwfam.iloc[-1]))
    noisy = ~group.description.isin(CLEAN)
    features[f"{prefix}_frac_noisy"] = noisy.mean()
    features[f"{prefix}_noisy_last3"] = noisy.tail(3).sum()
    features[f"{prefix}_n_noisy"] = noisy.sum()
    generic = group.description.isin(GENERIC)
    features[f"{prefix}_frac_gen_exact"] = generic.mean()
    main_mcc = FAMILY_MAIN_MCC.get(prefix)
    if main_mcc is not None:
        swapped = group.mcc.values != main_mcc
        features[f"{prefix}_swap_frac"] = swapped.mean()
        features[f"{prefix}_swap_last1"] = float(swapped[-1])
        features[f"{prefix}_swap_last3"] = swapped[-3:].sum()
        features[f"{prefix}_n_swap"] = swapped.sum()
        fee_rate = (group.fee.values > 0).mean()
        features[f"{prefix}_noise_score"] = swapped.mean() + noisy.mean() + generic.mean() + fee_rate
        features[f"{prefix}_n_clean"] = ((~swapped) & (~noisy.values) & (~generic.values)).sum()
    features[f"{prefix}_n_mcc"] = group.mcc.nunique()
    features[f"{prefix}_n_desc"] = group.description.nunique()
    features[f"{prefix}_fee_frac"] = (group.fee > 0).mean()
    features[f"{prefix}_fee_last"] = float(group.fee.iloc[-1] > 0)
    features[f"{prefix}_n_cur"] = group.currency.nunique()
    features[f"{prefix}_cur_last_diff"] = float(group.currency.iloc[-1] != group.currency.mode().iloc[0])
    log_amounts = np.log(amounts)
    median = np.median(log_amounts)
    features[f"{prefix}_logamt"] = median
    features[f"{prefix}_amt_cv"] = np.std(log_amounts)
    features[f"{prefix}_amt_range"] = log_amounts.max() - log_amounts.min()
    features[f"{prefix}_last_amt_dev"] = log_amounts[-1] - median
    features[f"{prefix}_last_amt_absdev"] = abs(log_amounts[-1] - median)
    features[f"{prefix}_max_absdev"] = np.max(np.abs(log_amounts - median))
    features[f"{prefix}_n_outlier"] = (np.abs(log_amounts - median) > 0.03).sum()
    if count >= 3:
        features[f"{prefix}_amt_slope"] = np.polyfit(days, log_amounts, 1)[0] * 30
    for window in (30, 45, 60, 90, 180):
        features[f"{prefix}_n{window}"] = (days > -window).sum()
    if count >= 2:
        median_gap = np.median(gaps)
        features[f"{prefix}_medgap"] = median_gap
        features[f"{prefix}_meangap"] = gaps.mean()
        features[f"{prefix}_gap_std"] = gaps.std()
        features[f"{prefix}_min_gap"] = gaps.min()
        features[f"{prefix}_max_gap"] = gaps.max()
        features[f"{prefix}_last_gap"] = gaps[-1]
        features[f"{prefix}_last_gap_ratio"] = gaps[-1] / median_gap if median_gap > 0 else np.nan
        features[f"{prefix}_due"] = days[-1] + median_gap
        features[f"{prefix}_overdue"] = (-days[-1]) / median_gap if median_gap > 0 else np.nan
        features[f"{prefix}_due_mean"] = days[-1] + gaps.mean()
        features[f"{prefix}_n_dup"] = (gaps < 5).sum()
        features[f"{prefix}_n_missed"] = (gaps > 1.6 * median_gap).sum()
        features[f"{prefix}_reg"] = (np.abs(gaps / median_gap - 1) < 0.2).mean()
        features[f"{prefix}_cov90"] = (days > -90).sum() * median_gap / 90.0
        features[f"{prefix}_dom_std"] = np.std(group.ts.dt.day.values)
    if refunds is not None and len(refunds):
        matched = refunds[
            (np.abs(np.log(refunds.amount.values / np.exp(median))) < 0.1) | (refunds.kwfam.values == prefix)
        ]
        features[f"{prefix}_nref"] = len(matched)
        features[f"{prefix}_ref_last"] = matched.day.max() if len(matched) else -999
        features[f"{prefix}_ref_after_last"] = float((matched.day > days[-1]).any()) if len(matched) else 0.0
        features[f"{prefix}_nref60"] = (matched.day > -60).sum()
        features[f"{prefix}_nref_frac"] = len(matched) / count
    else:
        features[f"{prefix}_nref"] = 0
        features[f"{prefix}_ref_last"] = -999
        features[f"{prefix}_ref_after_last"] = 0.0
        features[f"{prefix}_nref60"] = 0
        features[f"{prefix}_nref_frac"] = 0.0
    return features


def client_features(group: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {"n_txn": len(group), "first_txn": group.day.min(), "last_txn": group.day.max()}
    for transaction_type in ("card_payment", "topup", "p2p_transfer", "transfer", "atm", "refund", "fee"):
        features[f"n_{transaction_type}"] = (group.type == transaction_type).sum()
    features["n_in"] = (group.direction == "in").sum()
    features["n_cur"] = group.currency.nunique()
    features["main_cur"] = CURRENCIES.index(group.currency.value_counts().index[0])
    features["n_noise"] = group.isnoise.sum()
    features["n_nonrec"] = group.isnonrec.sum()
    features["fee_sum"] = group.fee.sum()
    features["n_feepos"] = (group.fee > 0).sum()
    features["salary_sum"] = group[group.description == "salary"].amount.sum()
    features["out_sum"] = group[group.direction == "out"].amount.sum()
    refunds = group[group.type == "refund"]
    for window in (30, 60, 90, 180):
        features[f"n_ref{window}"] = (refunds.day > -window).sum()
        features[f"n_txn{window}"] = (group.day > -window).sum()
    features["ref_rec_named"] = refunds.kwfam.notna().sum()
    features["ref_generic"] = refunds.description.isin(GENERIC).sum()
    features["ref_nonrec"] = refunds.isnonrec.sum()
    return features


def build_wide_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """One wide row per client with per-family blocks and cross-family ranks."""

    frame = prepare_transactions(transactions).sort_values(["client_id", "day"]).reset_index(drop=True)
    frame["cand"] = (frame.direction == "out") & (frame.type == "card_payment") & ~frame.isnoise & ~frame.isnonrec
    rows: dict[str, dict[str, float]] = {}
    for client_id, group in frame.groupby("client_id", sort=False):
        features = client_features(group)
        candidates = group[group.cand]
        refunds = group[group.type == "refund"]
        family = assign_family(candidates) if len(candidates) else pd.Series(dtype=object)
        candidates = candidates.assign(fam=family.values)
        features["n_cand"] = len(candidates)
        features["n_amb"] = (candidates.fam == "amb5812").sum()
        features["n_unknown"] = (candidates.fam == "unknown").sum()
        features["n_fam_present"] = (
            candidates.fam.isin(FAMILIES).sum() and candidates[candidates.fam.isin(FAMILIES)].fam.nunique()
        )
        for prefix in (*FAMILIES, "amb5812", "unknown"):
            features.update(timeline_features(candidates[candidates.fam == prefix], prefix, refunds))
        active = [
            (prefix, features[f"{prefix}_due"], features[f"{prefix}_n"], features[f"{prefix}_last"])
            for prefix in FAMILIES
            if features.get(f"{prefix}_n", 0) >= 2
            and features[f"{prefix}_last"] > -75
            and not np.isnan(features.get(f"{prefix}_due", np.nan))
        ]
        features["n_active"] = len(active)
        if active:
            by_due = sorted(active, key=lambda item: item[1])
            by_count = sorted(active, key=lambda item: -item[2])
            by_last = sorted(active, key=lambda item: -item[3])
            features["min_due"] = by_due[0][1]
            features["max_n_active"] = by_count[0][2]
            features["max_last_active"] = by_last[0][3]
            for prefix in FAMILIES:
                features[f"{prefix}_due_rank"] = next((i for i, item in enumerate(by_due) if item[0] == prefix), -1)
                features[f"{prefix}_n_rank"] = next((i for i, item in enumerate(by_count) if item[0] == prefix), -1)
                features[f"{prefix}_last_rank"] = next((i for i, item in enumerate(by_last) if item[0] == prefix), -1)
                if features.get(f"{prefix}_n", 0) >= 2 and f"{prefix}_due" in features:
                    features[f"{prefix}_due_minus_min"] = features[f"{prefix}_due"] - by_due[0][1]
        rows[client_id] = features
    return pd.DataFrame.from_dict(rows, orient="index")


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Unpivot per-family blocks into rows with family-agnostic names plus other-family context."""

    family_columns = {family: [c for c in wide.columns if c.startswith(family + "_")] for family in FAMILIES}
    suffixes = sorted({c[len(family) + 1 :] for family in FAMILIES for c in family_columns[family]})
    client_columns = [c for c in wide.columns if not any(c.startswith(family + "_") for family in FAMILIES)]
    parts = []
    for family_index, family in enumerate(FAMILIES):
        columns = {}
        for suffix in suffixes:
            name = f"{family}_{suffix}"
            columns["f_" + suffix] = wide[name] if name in wide.columns else pd.Series(np.nan, index=wide.index)
        for column in client_columns:
            columns[column] = wide[column]
        part = pd.DataFrame(columns, index=wide.index)
        others = [other for other in FAMILIES if other != family]
        other_counts = wide[[f"{other}_n" for other in others]].fillna(0)
        part["oth_max_n"] = other_counts.max(axis=1)
        part["oth_sum_n"] = other_counts.sum(axis=1)
        part["oth_n_active"] = (wide[[f"{other}_last" for other in others]] > -60).sum(axis=1)
        other_recent = wide[[f"{other}_last" for other in others]].values > -75
        other_due = wide[[f"{other}_due" for other in others]].where(other_recent)
        part["oth_min_due"] = other_due.min(axis=1)
        own_due = wide[f"{family}_due"] if f"{family}_due" in wide.columns else np.nan
        part["due_minus_oth_min"] = own_due - other_due.min(axis=1)
        part["oth_max_n90"] = wide[[f"{other}_n90" for other in others]].fillna(0).max(axis=1)
        part["fam_id"] = family_index
        part["client_id"] = wide.index
        part["fam"] = family
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def feature_columns(long: pd.DataFrame) -> list[str]:
    return [c for c in long.columns if c not in ("client_id", "fam", "y")]


def stage_two_inputs(scores: pd.DataFrame) -> np.ndarray:
    clipped = np.clip(scores[list(FAMILIES)].values, 1e-4, 1 - 1e-4)
    logits = np.log(clipped / (1 - clipped))
    ordered = np.sort(logits, axis=1)[:, ::-1]
    return np.hstack([logits, ordered[:, :2], (ordered[:, 0] - ordered[:, 1])[:, None]])


@dataclass
class KeywordStreamModel:
    boosters: list[lgb.Booster]
    columns: list[str]
    stage_two: LogisticRegression


def _fit_boosters(long: pd.DataFrame, columns: list[str], seeds: Sequence[int], rounds: int) -> list[lgb.Booster]:
    return [
        lgb.train({**PARAMETERS, "seed": seed}, lgb.Dataset(long[columns], long.y), num_boost_round=rounds)
        for seed in seeds
    ]


def _family_scores(boosters: Sequence[lgb.Booster], long: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scores = np.mean([booster.predict(long[columns]) for booster in boosters], axis=0)
    frame = pd.DataFrame({"client_id": long.client_id.values, "fam": long.fam.values, "s": scores})
    return frame.pivot(index="client_id", columns="fam", values="s")[list(FAMILIES)]


def fit(
    long: pd.DataFrame, labels: pd.Series, *, seeds: Sequence[int] = (0,), rounds: int = ROUNDS
) -> KeywordStreamModel:
    """Fit the bagged family scorer and the logistic second stage on inner out-of-fold scores."""

    long = long.assign(y=(long.client_id.map(labels) == long.fam).astype(float))
    columns = feature_columns(long)
    boosters = _fit_boosters(long, columns, seeds, rounds)
    inner_folds = long.client_id.map(lambda client_id: fold_of(INNER_FOLD_OFFSET + seeds[0], client_id)).values
    inner = np.zeros(len(long))
    for fold in range(5):
        booster = lgb.train(
            {**PARAMETERS, "seed": seeds[0]},
            lgb.Dataset(long.loc[inner_folds != fold, columns], long.y[inner_folds != fold]),
            num_boost_round=rounds,
        )
        inner[inner_folds == fold] = booster.predict(long.loc[inner_folds == fold, columns])
    inner_frame = pd.DataFrame({"client_id": long.client_id.values, "fam": long.fam.values, "s": inner})
    inner_scores = inner_frame.pivot(index="client_id", columns="fam", values="s")[list(FAMILIES)]
    stage_two = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    targets = labels.reindex(inner_scores.index).map(list(CLASSES).index).values
    stage_two.fit(stage_two_inputs(inner_scores), targets)
    return KeywordStreamModel(boosters=boosters, columns=columns, stage_two=stage_two)


def predict(model: KeywordStreamModel, long: pd.DataFrame) -> pd.DataFrame:
    scores = _family_scores(model.boosters, long, model.columns)
    probabilities = np.zeros((len(scores), len(CLASSES)))
    probabilities[:, model.stage_two.classes_] = model.stage_two.predict_proba(stage_two_inputs(scores))
    return pd.DataFrame(probabilities, index=scores.index, columns=list(CLASSES))
