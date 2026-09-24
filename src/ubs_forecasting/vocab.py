"""Closed description grammar and merchant-family evidence."""

from __future__ import annotations

import collections
from functools import cache

FAMILIES = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming")
CLASSES = (*FAMILIES, "none")
FAMILY_INDEX = {family: index for index, family in enumerate(FAMILIES)}
FAMILY_NAMES = {
    "cloud": ("cloud access", "cloud backup", "service plan", "storage plan"),
    "gym": ("urban gym", "fitness monthly", "fit club", "gym membership"),
    "insurance": ("cover plan", "policy premium", "safe cover", "insurance monthly", "cover monthly"),
    "mobile": ("phone contract", "service bill", "monthly plan", "digital plus"),
    "music": ("member pass", "audio streaming", "digital plus", "premium plan"),
    "software": ("saas billing", "software access", "productivity suite", "premium plan"),
    "streaming": ("media streaming", "video access", "digital plus", "premium plan"),
}
FAMILY_MCC = {
    "mobile": "4814",
    "cloud": "5732",
    "software": "5734",
    "insurance": "6300",
    "gym": "7997",
    "streaming": "5812",
    "music": "5812",
}
GENERIC_SUBSCRIPTION = ("subscription charge", "monthly plan", "member plan", "digital service")
GENERIC_EVERYDAY = ("digital order", "service payment", "card purchase", "merchant charge")
EVERYDAY = (
    "ride share",
    "fresh foods",
    "neighborhood market",
    "grocery store",
    "electronics shop",
    "online marketplace",
    "coffee shop",
    "casual dining",
    "pharmacy",
    "hotel booking",
)
OTHER = ("atm withdrawal", "salary", "p2p send", "p2p receive", "service fee")
ABBREVIATION_EXPANSIONS = {
    "dgtl": "digital",
    "prem": "premium",
    "mth": "monthly",
    "prod": "productivity",
    "stream": "streaming",
}
ABBREVIATIONS = {value: key for key, value in ABBREVIATION_EXPANSIONS.items()}
PREFIXES = frozenset({"member", "pay", "billing"})
SUFFIXES = frozenset({"core", "online", "service", "plus", "digital", "dgtl"})
CARD_MCCS = ("4111", "4814", "5411", "5732", "5734", "5812", "5912", "6300", "7011", "7997")

ALL_BASES = frozenset(name for names in FAMILY_NAMES.values() for name in names) | frozenset(
    GENERIC_SUBSCRIPTION + GENERIC_EVERYDAY + EVERYDAY + OTHER
)

TRUNCATIONS: dict[str, set[str]] = collections.defaultdict(set)
for base in ALL_BASES:
    tokens = base.split()
    if len(tokens) > 1:
        TRUNCATIONS[tokens[0]].add(base)
        TRUNCATIONS[tokens[-1]].add(base)

BASE_FAMILIES: dict[str, set[str]] = collections.defaultdict(set)
for family, names in FAMILY_NAMES.items():
    for name in names:
        BASE_FAMILIES[name].add(family)

KIND_PRIORITY = {"fam": 4, "generic_sub": 3, "generic_eve": 2, "everyday": 1, "other": 0}


@cache
def parse_description(description: str) -> tuple[frozenset[str], int, bool]:
    """Parse a description into base names, stripped-token count, and exactness.

    The tie-breaks and set iteration are explicitly sorted so output does not depend on
    ``PYTHONHASHSEED``.
    """

    tokens = description.split()
    candidates: list[tuple[int, int, frozenset[str]]] = []
    for prefix_count in range(0, min(3, len(tokens))):
        if prefix_count and tokens[prefix_count - 1] not in PREFIXES:
            break
        for suffix_count in range(0, min(3, len(tokens) - prefix_count)):
            if suffix_count and tokens[len(tokens) - suffix_count] not in SUFFIXES:
                break
            end = len(tokens) - suffix_count if suffix_count else len(tokens)
            core = [ABBREVIATION_EXPANSIONS.get(token, token) for token in tokens[prefix_count:end]]
            candidate = " ".join(core)
            if candidate in ALL_BASES:
                candidates.append((0, prefix_count + suffix_count, frozenset({candidate})))
            elif len(core) == 1 and candidate in TRUNCATIONS:
                candidates.append((1, prefix_count + suffix_count, frozenset(TRUNCATIONS[candidate])))
    if candidates:
        # Python's sort is stable; the prefix/suffix loop order is the grammar tie-break.
        candidates.sort(key=lambda item: (item[0], item[1]))
        match_type, stripped, bases = candidates[0]
        return bases, stripped, match_type == 0

    bases: set[str] = set()
    for token in tokens:
        token = ABBREVIATION_EXPANSIONS.get(token, token)
        if token in TRUNCATIONS and token not in PREFIXES | SUFFIXES:
            bases.update(TRUNCATIONS[token])
    return frozenset(bases), 99, False


@cache
def description_evidence(description: str) -> tuple[str, dict[str, float]]:
    """Return the grammar kind and normalized family weights for a description."""

    bases, _, _ = parse_description(description)
    if not bases:
        return "unknown", {}
    kinds: collections.Counter[str] = collections.Counter()
    family_weights: collections.Counter[str] = collections.Counter()
    for base in sorted(bases):
        weight = 1.0 / len(bases)
        if base in BASE_FAMILIES:
            kinds["fam"] += weight
            for family in sorted(BASE_FAMILIES[base]):
                family_weights[family] += weight / len(BASE_FAMILIES[base])
        if base in GENERIC_SUBSCRIPTION:
            kinds["generic_sub"] += weight
        if base in GENERIC_EVERYDAY:
            kinds["generic_eve"] += weight
        if base in EVERYDAY:
            kinds["everyday"] += weight
        if base in OTHER:
            kinds["other"] += weight
    total = sum(family_weights.values())
    kind = max(kinds, key=lambda value: (round(kinds[value], 9), KIND_PRIORITY[value]))
    weights = {family: value / total for family, value in sorted(family_weights.items())} if total else {}
    return kind, weights


# Compatibility names make comparison with the verified monolith straightforward.
parse = parse_description
evidence = description_evidence
