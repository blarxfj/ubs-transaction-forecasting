"""Label-independent recent activity and refund chronology features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def activity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only transaction types, directions, dates, amounts, and fees.

    These features are invariant to description/MCC corruption and are shared
    across every augmented view of a client.
    """
    day = (frame.date - pd.Timestamp("2026-01-01")).dt.days
    data = frame.assign(_day=day)
    rows = []
    for client, group in data.groupby("client_id", sort=True):
        row = {"client_id": client, "c_last_activity": group._day.max()}
        for name, mask in (
            ("all", np.ones(len(group), dtype=bool)),
            ("card", group.type == "card_payment"),
            ("refund", group.type == "refund"),
            ("incoming", group.direction == "in"),
        ):
            events = group[mask]
            row[f"c_{name}_last"] = events._day.max()
            for window in (30, 60, 90):
                recent = events[events._day >= -window]
                earlier = events[(events._day < -window) & (events._day >= -2 * window)]
                row[f"c_{name}_count{window}"] = len(recent)
                row[f"c_{name}_change{window}"] = (len(recent) + 1) / (len(earlier) + 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("client_id")


def attach_activity(long: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    """Add chronology, recent cessation, and candidate-relative refund features."""
    result = long.join(activity, on="client_id", validate="many_to_one")
    result["refund_after_last_payment"] = result.p_last_ref - result.p_last
    result["refunded_active_mass"] = result.p_prob * result.p_active * result.p_last_refunded
    result["refund_recency_cycles"] = -result.p_last_ref / result.p_per.clip(lower=1)
    result["payment_share_of_lifetime"] = result.p_span / (-result.c_first_day).clip(lower=1)
    return result
