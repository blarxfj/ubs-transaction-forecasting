"""Description, MCC, refund, and amount evidence for stream families."""

from __future__ import annotations

import collections
from typing import TypedDict

import numpy as np
import pandas as pd

from .vocab import FAMILIES, FAMILY_MCC, description_evidence


class AmountPrior(TypedDict):
    bins: np.ndarray
    logp: dict[str, np.ndarray]


type EvidenceMap = dict[str, tuple[str, dict[str, float]]]

EPSILON_DESCRIPTION = 0.10
MCC_NOISE = 0.30
EPSILON_REFUND = 0.05


def amount_log_probabilities(amount: float, prior: AmountPrior) -> np.ndarray:
    """Look up the family log probabilities for an amount."""

    bins = prior["bins"]
    bin_index = int(np.clip(np.searchsorted(bins, np.log(amount)) - 1, 0, len(bins) - 2))
    return np.array([prior["logp"][family][bin_index] for family in FAMILIES])


def description_log_likelihood(weights: np.ndarray, epsilon: float) -> np.ndarray:
    """Combine description evidence across stream rows."""

    informative = weights.sum(axis=1) > 0
    if not informative.any():
        return np.zeros(len(FAMILIES))
    likelihood = np.log((1 - epsilon) * weights[informative] + epsilon / len(FAMILIES)).sum(axis=0)
    return likelihood - likelihood.max()


def mcc_log_likelihood(mccs: np.ndarray) -> np.ndarray:
    """Combine merchant-category evidence across stream rows."""

    likelihood = np.zeros(len(FAMILIES))
    for mcc in mccs:
        likelihood += np.log(
            np.array([(1 - MCC_NOISE) * (mcc == FAMILY_MCC[family]) + MCC_NOISE / 10 for family in FAMILIES])
        )
    return likelihood - likelihood.max()


def learn_amount_prior(unlabeled: pd.DataFrame) -> AmountPrior:
    """Learn per-family amount histograms from clean recurring streams."""

    card_payments = unlabeled[unlabeled.type == "card_payment"]
    evidence: EvidenceMap = {
        description: description_evidence(description) for description in card_payments.description.unique()
    }
    kinds = card_payments.description.map(lambda description: evidence[description][0])
    card_payments = card_payments[kinds.isin(["fam", "generic_sub"])]
    amounts: dict[str, list[float]] = collections.defaultdict(list)
    for _, group in card_payments.groupby("client_id"):
        group = group.sort_values("amount")
        log_amounts = np.log(group.amount.values)
        split_points = np.where(np.diff(log_amounts) > 0.035)[0] + 1
        for indices in np.split(np.arange(len(group)), split_points):
            stream = group.iloc[indices]
            if len(stream) < 3:
                continue
            family_counts = collections.Counter(
                next(iter(evidence[description][1]))
                for description in stream.description
                if len(evidence[description][1]) == 1
            )
            if not family_counts:
                continue
            family, count = family_counts.most_common(1)[0]
            if count >= 2 and count >= 0.6 * len(stream):
                amounts[family].append(float(np.median(stream.amount)))
    bins = np.linspace(np.log(1.5), np.log(800), 41)
    log_probabilities: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        histogram, _ = np.histogram(np.log(amounts[family]), bins=bins)
        smoothed = (histogram + 0.5) / (histogram + 0.5).sum()
        log_probabilities[family] = np.log(smoothed)
    return {"bins": bins, "logp": log_probabilities}
