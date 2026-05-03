"""STL Decomposition detector.

Decomposes each per-group daily series into trend + seasonal + residual,
then flags days whose residual exceeds ``threshold`` standard deviations.
Strong on seasonal data and gradual drift (trend component).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def detect(
    long_df: pd.DataFrame,
    period: int = 7,
    threshold: float = 3.0,
    group_keys: Sequence[str] = ("service",),
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, *``group_keys``, ``cost``.
        period: seasonal period (7 = weekly).
        threshold: |residual / sigma| above this is flagged.
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
        for k, v in zip(keys, group_vals):
            sub[k] = v
        series = sub["cost"].to_numpy(dtype=float)
        if len(series) < 2 * period + 1:
            sub["score"] = 0.0
            sub["is_anomaly"] = False
            out.append(sub[["date", *keys, "cost", "score", "is_anomaly"]])
            continue

        try:
            stl = STL(pd.Series(series), period=period, robust=True).fit()
            resid = stl.resid.to_numpy()
            trend = stl.trend.to_numpy()
        except Exception:
            sub["score"] = 0.0
            sub["is_anomaly"] = False
            out.append(sub[["date", *keys, "cost", "score", "is_anomaly"]])
            continue

        sigma = np.std(resid) or 1.0
        resid_score = np.abs(resid) / sigma

        # Trend deviation captures gradual drift: how far the trend has moved
        # from its early-window baseline relative to the residual scale.
        baseline = np.median(trend[: max(period * 2, 7)])
        trend_dev = (trend - baseline) / max(baseline * 0.1, sigma)
        trend_score = np.clip(trend_dev, 0.0, None)

        score = np.maximum(resid_score, trend_score)
        sub["score"] = score
        sub["is_anomaly"] = sub["score"] >= threshold
        out.append(sub[["date", *keys, "cost", "score", "is_anomaly"]])

    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["date", *keys])
        .reset_index(drop=True)
    )
