"""Isolation Forest detector.

Trains one IsolationForest per service on multivariate features:
    - daily cost
    - 7-day rolling mean
    - 7-day rolling std
    - day-over-day pct change
    - day of week (cyclic encoding)

Anomaly score = ``-decision_function`` (higher = more anomalous).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _features(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("date").copy()
    sub["roll_mean"] = sub["cost"].rolling(7, min_periods=2).mean()
    sub["roll_std"] = sub["cost"].rolling(7, min_periods=2).std().fillna(0.0)
    sub["pct_change"] = sub["cost"].pct_change().fillna(0.0)
    dow = sub["date"].dt.weekday
    sub["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    sub["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    sub = sub.bfill()
    return sub


def detect(
    long_df: pd.DataFrame,
    contamination: float = 0.05,
    threshold_quantile: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, ``service``, ``cost``.
        contamination: expected anomaly fraction (passed to IsolationForest).
        threshold_quantile: anomaly-score quantile cutoff for ``is_anomaly``.
    """
    out = []
    feature_cols = ["cost", "roll_mean", "roll_std", "pct_change", "dow_sin", "dow_cos"]

    for service, sub in long_df.groupby("service"):
        feats = _features(sub)
        X = feats[feature_cols].to_numpy()
        if len(X) < 10:
            feats["score"] = 0.0
            feats["is_anomaly"] = False
            out.append(feats[["date", "service", "cost", "score", "is_anomaly"]])
            continue

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(X)
        # Higher decision_function = more normal; flip sign so higher = anomalous.
        raw = -model.decision_function(X)
        # Min-max normalize to [0, 1] for comparability with other detectors.
        rng = raw.max() - raw.min()
        score = (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)
        cutoff = np.quantile(score, threshold_quantile)
        feats["score"] = score
        feats["is_anomaly"] = score >= cutoff
        feats["service"] = service
        out.append(feats[["date", "service", "cost", "score", "is_anomaly"]])

    return pd.concat(out, ignore_index=True).sort_values(["date", "service"]).reset_index(drop=True)
