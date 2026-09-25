"""Listwise candidate component: eight candidate rows per client scored by one shared ranker.

Families are reconstructed from MCC, description keywords, and same-client similar-amount
evidence propagated through an amount kernel. Every client becomes eight candidate rows (seven
families plus ``none``) that carry the family's own stream statistics, its ranks and deltas
against the other families, client-level context, and the attributes of the earliest-due
competitors. A symmetric-tree ranker trained with a per-client softmax loss scores the rows; the
``none`` probability is multiplied by a fixed factor before the argmax.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from ..vocab import CLASSES

CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
MCC_FAMILY = {5732: 0, 7997: 1, 6300: 2, 4814: 3, 5734: 5}
KEYWORDS = [
    r"cloud|backup|storage",
    r"gym|fitness|\bfit\b|urban|club|membership",
    r"insurance|policy|cover|safe",
    r"phone|contract|service bill",
    r"audio|member pass|\bpass\b",
    r"software|saas|productivity|\bprod\b|suite",
    r"video|media",
]
BACKGROUND = re.compile(
    r"salary|atm|fresh foods|pharmacy|hotel|booking|electronics|ride|coffee|grocery|neighborhood|marketplace|"
    r"casual|dining|p2p|service fee"
)
GENERIC = ("monthly plan", "member plan", "subscription charge", "digital service")
VAGUE = ("card purchase", "digital order", "merchant charge", "service payment")
JOINT_NAMES = [
    "n",
    "age",
    "amount_cv",
    "gap_med",
    "core_n",
    "core_age",
    "core_amount_cv",
    "core_cadence_due",
    "core_cadence_error",
    "cadence_due",
    "cadence_error",
    "core_active_due_1.5",
    "refund_n",
]
PERIOD_GRID = np.array([7.0, 14.0, 30.0, 30.4375, 60.0, 60.875, 90.0, 91.3125, 180.0, 365.0])
NONE_WEIGHT = 0.8
RANKER_PARAMETERS = {
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.04,
    "l2_leaf_reg": 5,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "QuerySoftMax",
}


def assign_families(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Combine MCC and text evidence with same-client similar-amount evidence."""

    count = len(group)
    evidence = np.zeros((count, 7))
    eligible = np.ones(count, bool)
    for index, row in enumerate(group.itertuples()):
        description = str(row.description).lower()
        if BACKGROUND.search(description) or row.type not in ("card_payment", "refund"):
            eligible[index] = False
            continue
        if row.mcc in MCC_FAMILY:
            evidence[index, MCC_FAMILY[row.mcc]] += 1.0
        if row.mcc == 5812:
            evidence[index, [4, 6]] += 0.45
        for family, pattern in enumerate(KEYWORDS):
            if re.search(pattern, description):
                evidence[index, family] += 1.4
        if "service plan" in description:
            evidence[index, 0] += 1.0
        if "monthly plan" in description and row.mcc not in MCC_FAMILY and row.mcc != 5812:
            evidence[index, 3] += 0.2
    amount = np.maximum(group.amount.to_numpy(), 0.1)
    distance = abs(np.log(amount[:, None] / amount[None, :]))
    kernel = np.exp(-0.5 * (distance / 0.035) ** 2) * (distance < 0.12) * eligible[:, None] * eligible[None, :]
    smooth = kernel @ evidence
    score = evidence + 1.3 * smooth / np.maximum(kernel.sum(axis=1, keepdims=True), 1)
    family = score.argmax(axis=1)
    family[~eligible] = -1
    confidence = score.max(axis=1)
    family[confidence < 0.05] = -1
    return family, confidence


def stream_statistics(days: np.ndarray, amounts: np.ndarray) -> dict[str, float]:
    """Recency, counts, gaps, amount dispersion, phase, and latent-cadence reconstruction."""

    days, amounts = np.asarray(days), np.asarray(amounts)
    out: dict[str, float] = {"n": len(days)}
    if not len(days):
        return out
    order = np.argsort(days)
    days = days[order]
    amounts = amounts[order]
    age = -days[-1]
    mean_amount = max(amounts.mean(), 0.01)
    out.update(
        age=age,
        first_age=-days[0],
        span=days[-1] - days[0],
        amount_mean=amounts.mean(),
        amount_last=amounts[-1],
        amount_med=np.median(amounts),
        amount_cv=amounts.std() / mean_amount,
        amount_mad=np.median(abs(amounts - np.median(amounts))) / max(np.median(amounts), 0.01),
        amount_range=(amounts.max() - amounts.min()) / mean_amount,
    )
    for window in (15, 30, 45, 60, 90, 120, 180, 270):
        out[f"n{window}"] = int((days >= -window).sum())
    for j in range(1, 6):
        out[f"age{j}"] = -days[-j] if len(days) >= j else np.nan
    if len(days) >= 2:
        gap = np.diff(days)
        out.update(
            gap_mean=gap.mean(),
            gap_med=np.median(gap),
            gap_std=gap.std(),
            gap_min=gap.min(),
            gap_max=gap.max(),
            gap_last=gap[-1],
            gap_recent=np.median(gap[-3:]),
            amount_delta=(amounts[-1] - amounts[0]) / mean_amount,
            amount_last_delta=(amounts[-1] - amounts[-2]) / mean_amount,
        )
        out["amount_slope"] = np.polyfit(days - days.mean(), amounts / mean_amount, 1)[0]
        for name in ("gap_mean", "gap_med", "gap_recent"):
            out["due_" + name] = out[name] - age
            out["overdue_" + name] = age / max(out[name], 1)
        for j in range(1, 5):
            out[f"gap{j}"] = gap[-j] if len(gap) >= j else np.nan
        out["regular_amount"] = float(np.mean(abs(np.diff(amounts)) / np.maximum(amounts[:-1], 0.01) < 0.035))
    for period in (7, 14, 30, 60, 90, 365):
        angle = days * 2 * np.pi / period
        phase = np.mean(np.exp(1j * angle))
        out[f"phase_strength{period}"] = abs(phase)
        out[f"phase_due{period}"] = (np.angle(phase) / (2 * np.pi) * period) % period
        out[f"last_due{period}"] = period - age
        if len(days) >= 2:
            gaps = np.diff(days)
            rounds = np.maximum(np.round(gaps / period), 1)
            out[f"gap_error{period}"] = np.mean(np.minimum(abs(gaps - rounds * period), period)) / period
    if len(days) >= 2:
        gaps = np.diff(days)
        costs = []
        for period in PERIOD_GRID:
            steps = np.maximum(np.round(gaps / period), 1)
            costs.append(np.median(abs(gaps / steps - period)) / period + 0.12 * np.mean(steps - 1))
        period = PERIOD_GRID[int(np.argmin(costs))]
        steps = np.maximum(np.round(gaps / period), 1)
        index = np.r_[0, np.cumsum(steps)]
        slope = (days[-1] - days[0]) / max(index[-1], 1)
        residual = days - index * period
        next_phase = np.median(residual[-5:]) + (index[-1] + 1) * period
        out.update(
            cadence=period,
            cadence_error=min(costs),
            cadence_missing=float(np.sum(steps - 1)),
            cadence_missing_fraction=float(np.mean(steps - 1)),
            cadence_slope=slope,
            cadence_due=next_phase,
            cadence_last_due=days[-1] + period,
            cadence_active_due=next_phase if next_phase >= -7 else 999,
            cadence_age_ratio=age / period,
        )
        for size in (3, 5, 8):
            recent_days = days[-size:]
            recent_index = index[-size:]
            fit = np.polyfit(recent_index - recent_index[-1], recent_days, 1)
            out[f"linear_due{size}"] = float(fit[0] + fit[1])
            out[f"linear_period{size}"] = float(fit[0])
            out[f"linear_error{size}"] = float(np.std(recent_days - np.polyval(fit, recent_index - recent_index[-1])))
        for multiple in (1.2, 1.5, 2.0):
            out[f"active_n_{multiple}"] = len(days) if age < period * multiple else 0
            out[f"active_due_{multiple}"] = max(next_phase, 0) if age < period * multiple else 999
    return out


def client_profile(group: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Client-level context plus one statistics block per family."""

    group = group.sort_values("day").reset_index(drop=True)
    family, confidence = assign_families(group)
    group["family"] = family
    out: dict[str, float] = {
        "total_n": len(group),
        "history_age": -group.day.min(),
        "all_age": -group.day.max(),
        "n_refunds": int((group.type == "refund").sum()),
        "n_currencies": group.currency.nunique(),
    }
    generic = group.description.isin(GENERIC)
    vague = group.description.isin(VAGUE)
    out["generic_fraction"] = generic.mean()
    out["vague_fraction"] = vague.mean()
    out["assigned_fraction"] = (family >= 0).mean()
    group["generic"] = generic.to_numpy()
    group["vague"] = vague.to_numpy()
    per_family = []
    for k in range(7):
        payments = group[(group.family == k) & (group.type == "card_payment")]
        block = stream_statistics(payments.day.to_numpy(), payments.amount.to_numpy())
        refunds = group[(group.family == k) & (group.type == "refund")]
        block["refund_n"] = len(refunds)
        block["refund_age"] = -refunds.day.max() if len(refunds) else np.nan
        block["refund_n60"] = int((refunds.day >= -60).sum())
        if len(payments):
            amounts = payments.amount.to_numpy()
            days = payments.day.to_numpy()
            distance = abs(np.log(np.maximum(amounts[:, None], 0.1) / np.maximum(amounts[None, :], 0.1)))
            weights = (distance < 0.04).sum(axis=1)
            core = distance[np.argmax(weights)] < 0.04
            block.update({"core_" + key: value for key, value in stream_statistics(days[core], amounts[core]).items()})
            block["core_fraction"] = core.mean()
            block["text_confidence"] = float(confidence[payments.index].mean())
            block["generic_fraction"] = float(payments.generic.mean())
            block["vague_fraction"] = float(payments.vague.mean())
            block["specific_fraction"] = float(payments.description.str.contains(KEYWORDS[k], regex=True).mean())
            recent = np.flatnonzero(days >= -100)
            if len(recent):
                seed = recent[np.argmax(weights[recent])]
                subset = distance[seed] < 0.07
                recent_core = stream_statistics(days[subset], amounts[subset])
                block.update({"recentcore_" + key: value for key, value in recent_core.items()})
            for mcc in MCC_FAMILY:
                block[f"mcc_{mcc}"] = float((payments.mcc == mcc).mean())
        per_family.append(block)
    out["family_n"] = sum(block["n"] > 0 for block in per_family)
    out["active_family_n"] = sum(block.get("age", 999) < 45 for block in per_family)
    return out, per_family


@dataclass
class CandidateTensor:
    ids: np.ndarray
    tensor: np.ndarray
    globals_: np.ndarray
    features: list[str]
    global_keys: list[str]

    def subset(self, client_ids: Sequence[str]) -> CandidateTensor:
        positions = pd.Index(self.ids).get_indexer(list(client_ids))
        if (positions < 0).any():
            raise KeyError("unknown client identifiers requested from the candidate tensor")
        return CandidateTensor(
            self.ids[positions], self.tensor[positions], self.globals_[positions], self.features, self.global_keys
        )


def build_tensor(transactions: pd.DataFrame) -> CandidateTensor:
    """Compute the (client, candidate, feature) tensor for every client in the frame."""

    frame = transactions.copy()
    frame["day"] = (frame.timestamp - CUTOFF).dt.total_seconds() / 86400
    if not (frame.day < 0).all():
        raise ValueError("future transactions must not enter features")
    frame["mcc"] = pd.to_numeric(frame.mcc)
    rows = {client_id: client_profile(group) for client_id, group in frame.groupby("client_id", sort=True)}
    features = sorted({key for _, per_family in rows.values() for block in per_family for key in block})
    global_keys = sorted({key for context, _ in rows.values() for key in context})
    client_ids = sorted(rows)
    tensor = np.full((len(rows), 8, len(features)), np.nan)
    globals_ = np.zeros((len(rows), len(global_keys)))
    for i, client_id in enumerate(client_ids):
        context, per_family = rows[client_id]
        globals_[i] = [context[key] for key in global_keys]
        for k in range(7):
            tensor[i, k] = [per_family[k].get(key, np.nan) for key in features]
    return CandidateTensor(np.array(client_ids), tensor, globals_, features, global_keys)


def candidate_matrix(data: CandidateTensor) -> np.ndarray:
    """Shared candidate representation: own block, ranks, deltas, context, and competitors."""

    t = data.tensor
    g = data.globals_
    names = data.features
    valid = np.where(np.isnan(t[:, :7]), np.inf, t[:, :7])
    sorted_ = np.sort(valid, axis=1)
    sorted_[~np.isfinite(sorted_)] = np.nan
    maximum = np.nan_to_num(t[:, :7], nan=-999).max(axis=1)
    summary = np.concatenate([sorted_[:, :3].reshape(len(t), -1), maximum], axis=1)
    joint = t[:, :7, [names.index(name) for name in JOINT_NAMES]]
    criterion = np.nan_to_num(t[:, :7, names.index("core_active_due_1.5")], nan=999)
    ordering = np.argsort(criterion, axis=1, kind="stable")
    ordered = np.take_along_axis(joint, ordering[:, :, None], axis=1)
    summary = np.concatenate([summary, ordered[:, :4].reshape(len(t), -1)], axis=1)
    rank = np.sum(t[:, :, None, :] > t[:, None, :7, :], axis=2)
    delta = t - sorted_[:, 0:1, :]
    candidate = np.concatenate(
        [
            t,
            rank,
            delta,
            np.repeat(g[:, None, :], 8, axis=1),
            np.repeat(summary[:, None, :], 8, axis=1),
            np.tile(np.eye(8)[None, :, :], (len(t), 1, 1)),
        ],
        axis=2,
    )
    return np.nan_to_num(candidate, nan=-999, posinf=9999, neginf=-999)


def adjust(probabilities: np.ndarray, none_weight: float = NONE_WEIGHT) -> np.ndarray:
    adjusted = probabilities.copy()
    adjusted[:, 7] *= none_weight
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def fit(
    matrix: np.ndarray, labels: np.ndarray, *, seed: int = 0, parameters: dict[str, Any] | None = None
) -> CatBoostRanker:
    """Fit the listwise ranker; ``labels`` holds class indices for each client of ``matrix``."""

    target = (np.arange(8)[None, :] == labels[:, None]).astype(int)
    model = CatBoostRanker(**{**RANKER_PARAMETERS, **(parameters or {})}, random_seed=seed)
    pool = Pool(matrix.reshape(-1, matrix.shape[-1]), target.reshape(-1), group_id=np.repeat(np.arange(len(matrix)), 8))
    model.fit(pool)
    return model


def predict(model: CatBoostRanker, matrix: np.ndarray, client_ids: Sequence[str]) -> pd.DataFrame:
    raw = model.predict(matrix.reshape(-1, matrix.shape[-1])).reshape(-1, 8)
    probabilities = np.exp(raw - raw.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    index = pd.Index(list(client_ids), name="client_id")
    return pd.DataFrame(adjust(probabilities), index=index, columns=list(CLASSES))
