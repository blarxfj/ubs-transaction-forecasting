"""Self-supervised pseudo cutoffs: shifted histories labeled from each client's own observed future.

Every transaction history runs up to the real cutoff. Moving the cutoff back by ``days`` turns the
last ``days`` of a history into an observed future: the family of the earliest recurring event in
the 90-day horizon after the pseudo cutoff is exactly what the challenge label describes, and
``none`` when no recurring stream fires in that horizon. This yields labeled training clients
without touching any provided label, and the unlabeled pretrain histories are clean enough that
their observed futures give nearly noise-free targets.

Labels are soft: the target distribution over the eight classes is the seven-family posterior of
the stream that fires first, so an ambiguous future stream contributes fractional evidence instead
of a wrong hard label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CUTOFF, SUBSCRIPTION_KINDS, row_evidence
from .posterior import AmountPrior
from .streams import build_streams, stream_record
from .vocab import CLASSES, FAMILIES

HORIZON = 90


def shift_history(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Move the cutoff back by ``days``: drop the observed future and re-date the past to the cutoff.

    The result looks like an ordinary history ending at the real cutoff, so every downstream
    feature builder and re-noising step applies unchanged.
    """

    if days <= 0:
        raise ValueError("shift must be a positive number of days")
    pseudo_cutoff = CUTOFF - pd.Timedelta(days=days)
    past = frame[frame.date < pseudo_cutoff].copy()
    past["date"] = past.date + pd.Timedelta(days=days)
    past["timestamp"] = past.timestamp + pd.Timedelta(days=days)
    return past.reset_index(drop=True)


def pseudo_labels(frame: pd.DataFrame, amount_prior: AmountPrior, days: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Soft eight-class targets for every client from the observed window after a pseudo cutoff.

    Streams are reconstructed on the whole history (past and observed future) so a future event is
    recognized as recurring when its amount stream has at least two events overall. The label of a
    client is the posterior of the recurring stream whose first event in the horizon comes earliest;
    clients without such an event are labeled ``none``.
    """

    if days <= 0:
        raise ValueError("shift must be a positive number of days")
    enriched = row_evidence(frame)
    start = -days
    end = min(-days + horizon, 0)
    rows: dict[str, np.ndarray] = {}
    for client_id, group in enriched.groupby("client_id", sort=True):
        target = np.zeros(len(CLASSES))
        candidates = group[(group.type == "card_payment") & group.kind.isin(SUBSCRIPTION_KINDS)]
        refunds = group[group.type == "refund"]
        main_currency = group.currency.value_counts().index[0]
        earliest: tuple[int, np.ndarray] | None = None
        for stream in build_streams(candidates) if len(candidates) else []:
            if len(stream) < 2:
                continue
            future_days = stream.day.values[(stream.day.values >= start) & (stream.day.values < end)]
            if not len(future_days):
                continue
            first = int(future_days.min())
            if earliest is None or first < earliest[0]:
                record = stream_record(stream, refunds, main_currency, amount_prior)
                earliest = (first, record["post"])
        if earliest is None:
            target[len(FAMILIES)] = 1.0
        else:
            target[: len(FAMILIES)] = earliest[1]
        rows[client_id] = target
    output = pd.DataFrame.from_dict(rows, orient="index", columns=list(CLASSES))
    output.index.name = "client_id"
    return output
