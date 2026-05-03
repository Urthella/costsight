"""STL Decomposition detector.

Decomposes each per-service daily series into trend + seasonal + residual,
then flags days whose residual exceeds ``threshold`` standard deviations.
Strong on seasonal data and gradual drift (trend component).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def detect(
    long_df: pd.DataFrame,
    period: int = 7,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, ``service``, ``cost``.
        period: seasonal period (7 = weekly).
        threshold: |residual / sigma| above this is flagged.
    """
    out = []
    for service, sub in long_df.groupby("service"):
        sub = sub.sort_values("date").copy()
        series = sub["cost"].to_numpy(dtype=float)
        if len(series) < 2 * period + 1:
            sub["score"] = 0.0
            sub["is_anomaly"] = False
            out.append(sub[["date", "service", "cost", "score", "is_anomaly"]])
            continue

        try:
            stl = STL(pd.Series(series), period=period, robust=True).fit()
            resid = stl.resid.to_numpy()
            trend = stl.trend.to_numpy()
        except Exception:
            sub["score"] = 0.0
            sub["is_anomaly"] = False
            out.append(sub[["date", "service", "cost", "score", "is_anomaly"]])
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
        sub["service"] = service
        out.append(sub[["date", "service", "cost", "score", "is_anomaly"]])

    return pd.concat(out, ignore_index=True).sort_values(["date", "service"]).reset_index(drop=True)
