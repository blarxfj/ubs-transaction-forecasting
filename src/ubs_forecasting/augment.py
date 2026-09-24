"""Label-independent transaction re-noising for split-shift calibration."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .features import SUBSCRIPTION_KINDS, row_evidence
from .posterior import AmountPrior
from .streams import build_streams, stream_record
from .vocab import (
    ABBREVIATIONS,
    CARD_MCCS,
    FAMILIES,
    FAMILY_MCC,
    FAMILY_NAMES,
    GENERIC_EVERYDAY,
    GENERIC_SUBSCRIPTION,
    PREFIXES,
    SUFFIXES,
    description_evidence,
    parse_description,
)

TEST_LEVEL = {
    "r_mask": 0.54,
    "r_mcc": 0.22,
    "r_affix": 0.65,
    "r_eve_mask": 0.67,
    "r_eve_mcc": 0.12,
}
VALID_LEVEL = {
    "r_mask": 0.37,
    "r_mcc": 0.16,
    "r_affix": 0.59,
    "r_eve_mask": 0.47,
    "r_eve_mcc": 0.09,
}
VALID_TO_TEST = {
    "p_eve_mask": 0.38,
    "p_eve_mcc": 0.04,
    "p_sub_mask": 0.25,
    "p_sub_mcc": 0.10,
    "p_sub_affix": 0.08,
}


def add_affixes(description: str, rng: random.Random) -> str:
    """Apply a deterministic-seed draw from the observed affix grammar."""

    tokens = description.split()
    operation = rng.random()
    if operation < 0.3 and len(tokens) >= 2:
        tokens = [tokens[0]] if rng.random() < 0.5 else [tokens[-1]]
    elif operation < 0.5:
        tokens = [ABBREVIATIONS.get(token, token) if rng.random() < 0.7 else token for token in tokens]
    if rng.random() < 0.35:
        tokens = [rng.choice(sorted(PREFIXES)), *tokens]
    if rng.random() < 0.6:
        tokens = [*tokens, rng.choice(sorted(SUFFIXES))]
    return " ".join(tokens)


def redraw_noise(
    frame: pd.DataFrame,
    amount_prior: AmountPrior,
    r_mask: float,
    r_mcc: float,
    r_affix: float,
    r_eve_mask: float,
    r_eve_mcc: float,
    seed: int = 0,
    min_posterior: float = 0.6,
) -> pd.DataFrame:
    """Re-draw card-payment noise without consulting client labels."""

    rng = random.Random(seed)
    enriched = row_evidence(frame)
    family_by_index: dict[int, str] = {}
    for _, group in enriched.groupby("client_id"):
        candidates = group[(group.type == "card_payment") & group.kind.isin(SUBSCRIPTION_KINDS)]
        if candidates.empty:
            continue
        main_currency = group.currency.value_counts().index[0]
        refunds = group[group.type == "refund"]
        for stream in build_streams(candidates):
            record = stream_record(stream, refunds, main_currency, amount_prior)
            family_index = int(np.argmax(record["post"]))
            if len(stream) < 2 or record["post"][family_index] < min_posterior:
                # Preserve uncertainty for sparse/ambiguous candidates while still replacing
                # their train-specific description quality without consulting the label.
                family = rng.choices(FAMILIES, weights=record["post"], k=1)[0]
            else:
                family = FAMILIES[family_index]
            for index in stream.index:
                family_by_index[index] = family

    output = frame.copy()
    descriptions = output.description.to_dict()
    mccs = output.mcc.to_dict()
    kinds = enriched.kind.to_dict()
    for index in output.index[output.type == "card_payment"]:
        kind = kinds[index]
        if index in family_by_index:
            family = family_by_index[index]
            if rng.random() < r_mask:
                descriptions[index] = rng.choice(GENERIC_SUBSCRIPTION)
            else:
                original = descriptions[index]
                base = (
                    original
                    if kind == "fam" and family in description_evidence(original)[1]
                    else rng.choice(FAMILY_NAMES[family][:4])
                )
                bases = parse_description(base)[0]
                clean = next(iter(bases)) if len(bases) == 1 else base
                descriptions[index] = add_affixes(clean, rng) if rng.random() < r_affix else clean
            mccs[index] = rng.choice(CARD_MCCS) if rng.random() < r_mcc else FAMILY_MCC[family]
        elif kind == "everyday":
            if rng.random() < r_eve_mask:
                descriptions[index] = rng.choice(GENERIC_EVERYDAY)
            if rng.random() < r_eve_mcc:
                mccs[index] = rng.choice(CARD_MCCS)
    output["description"] = pd.Series(descriptions)
    output["mcc"] = pd.Series(mccs)
    return output


def add_noise(
    frame: pd.DataFrame,
    p_eve_mask: float,
    p_eve_mcc: float,
    p_sub_mask: float,
    p_sub_mcc: float,
    p_sub_affix: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Add enough noise to valid transactions to approximate test noise."""

    rng = random.Random(seed)
    output = frame.copy()
    kinds = output.description.map(
        {description: description_evidence(description)[0] for description in output.description.unique()}
    ).values
    descriptions = output.description.values.copy()
    mccs = output.mcc.values.copy()
    for index in np.where((output.type == "card_payment").values)[0]:
        if kinds[index] == "everyday":
            if rng.random() < p_eve_mask:
                descriptions[index] = rng.choice(GENERIC_EVERYDAY)
            if rng.random() < p_eve_mcc:
                mccs[index] = rng.choice(CARD_MCCS)
        elif kinds[index] in ("fam", "generic_sub"):
            masked = kinds[index] == "fam" and rng.random() < p_sub_mask
            if masked:
                descriptions[index] = rng.choice(GENERIC_SUBSCRIPTION)
            elif kinds[index] == "fam" and rng.random() < p_sub_affix:
                descriptions[index] = add_affixes(descriptions[index], rng)
            if rng.random() < p_sub_mcc:
                mccs[index] = rng.choice(CARD_MCCS)
    output["description"] = descriptions
    output["mcc"] = mccs
    return output
