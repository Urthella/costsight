"""Consensus / ensemble detector — combines the three base detectors.

Hard vote: a (date, service) point is flagged when at least `min_votes`
of the base detectors agree (default 2 of 3). The score is the max of
the normalized base scores, so existing severity / alerting logic keeps
working unchanged.
"""
from __future__ import annotations

import pandas as pd

from .iforest import detect as iforest_detect
from .stl import detect as stl_detect
from .zscore import detect as zscore_detect


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def detect(df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    base = {
        "zscore": zscore_detect(df),
        "stl": stl_detect(df),
        "iforest": iforest_detect(df),
    }
    merged = None
    for name, det in base.items():
        d = det[["date", "service", "cost", "score", "is_anomaly"]].copy()
        d[f"{name}_flag"] = d["is_anomaly"].astype(int)
        d[f"{name}_norm"] = _normalize(d["score"])
        d = d.drop(columns=["score", "is_anomaly"])
        if merged is None:
            merged = d
        else:
            merged = merged.merge(d, on=["date", "service", "cost"], how="outer")

    flag_cols = [c for c in merged.columns if c.endswith("_flag")]
    norm_cols = [c for c in merged.columns if c.endswith("_norm")]
    merged["votes"] = merged[flag_cols].sum(axis=1)
    merged["score"] = merged[norm_cols].max(axis=1)
    merged["is_anomaly"] = merged["votes"] >= min_votes

    return merged[["date", "service", "cost", "score", "is_anomaly"]].sort_values(
        ["date", "service"]
    ).reset_index(drop=True)
