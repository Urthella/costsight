"""Rolling Z-Score baseline detector.

A point is anomalous when its rolling z-score exceeds ``threshold``.
Run independently per group (default ``service``), on the daily
long-format frame.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def detect(
    long_df: pd.DataFrame,
    window: int = 14,
    threshold: float = 3.0,
    group_keys: Sequence[str] = ("service",),
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, *``group_keys``, ``cost``.
        window: rolling window size in days.
        threshold: |z| above this is flagged.
        group_keys: columns that identify an independent series. Defaults to
            ``("service",)`` for backward compatibility; pass
            ``("service", "env")`` for multi-granularity scoring.
    """
    keys = list(group_keys)
    out = []
    for group_vals, sub in long_df.groupby(keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        sub = sub.sort_values("date").copy()
        roll = sub["cost"].rolling(window=window, min_periods=max(window // 2, 3))
        mean = roll.mean()
        std = roll.std().replace(0.0, np.nan)
        z = (sub["cost"] - mean) / std
        z = z.fillna(0.0)
        sub["score"] = z.abs()
        sub["is_anomaly"] = sub["score"] >= threshold
        for k, v in zip(keys, group_vals):
            sub[k] = v
        out.append(sub[["date", *keys, "cost", "score", "is_anomaly"]])
    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["date", *keys])
        .reset_index(drop=True)
    )
