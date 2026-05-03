"""Rolling Z-Score baseline detector.

A point is anomalous when its rolling z-score exceeds ``threshold``.
Run independently per service, on the daily long-format frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect(
    long_df: pd.DataFrame,
    window: int = 14,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, ``service``, ``cost``.
        window: rolling window size in days.
        threshold: |z| above this is flagged.
    """
    out = []
    for service, sub in long_df.groupby("service"):
        sub = sub.sort_values("date").copy()
        roll = sub["cost"].rolling(window=window, min_periods=max(window // 2, 3))
        mean = roll.mean()
        std = roll.std().replace(0.0, np.nan)
        z = (sub["cost"] - mean) / std
        z = z.fillna(0.0)
        sub["score"] = z.abs()
        sub["is_anomaly"] = sub["score"] >= threshold
        sub["service"] = service
        out.append(sub[["date", "service", "cost", "score", "is_anomaly"]])
    return pd.concat(out, ignore_index=True).sort_values(["date", "service"]).reset_index(drop=True)
