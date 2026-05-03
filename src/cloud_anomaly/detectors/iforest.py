"""Isolation Forest detector.

Trains one IsolationForest per group (default: per service) on a
multivariate feature set engineered to expose all three target anomaly
types:

    - cost / log_cost           raw + log-scale spend
    - rel_to_median / rel_mean  cost normalized vs. recent baselines
    - roll_std_14               rolling std (level-shift signature)
    - pct_change                day-over-day change (sudden moves)
    - trend_dev / trend_slope   30-day deviation + slope (drift signature)
    - lag_1 / lag_7             yesterday + same-weekday-last-week ratios
    - season_resid              residual after removing weekly seasonality
    - dow_sin / dow_cos         cyclic day-of-week encoding

Anomalies are flagged via the model's native ``predict`` (driven by
``contamination``); the per-service min-max-normalized
``-decision_function`` is exposed as ``score``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLS = [
    "cost",
    "log_cost",
    "rel_to_median",
    "rel_to_mean",
    "roll_std_14",
    "pct_change",
    "trend_dev",
    "trend_slope",
    "lag_1_ratio",
    "lag_7_ratio",
    "season_resid",
    "dow_sin",
    "dow_cos",
]


def _features(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("date").copy()
    cost = sub["cost"].astype(float)

    roll_median_14 = cost.rolling(14, min_periods=3).median()
    roll_mean_14 = cost.rolling(14, min_periods=3).mean()
    roll_median_30 = cost.rolling(30, min_periods=3).median()

    sub["log_cost"] = np.log1p(cost)
    sub["rel_to_median"] = (cost / roll_median_14.replace(0, np.nan)).fillna(1.0)
    sub["rel_to_mean"] = (cost / roll_mean_14.replace(0, np.nan)).fillna(1.0)
    sub["roll_std_14"] = cost.rolling(14, min_periods=3).std().fillna(0.0)
    sub["pct_change"] = cost.pct_change().fillna(0.0)
    sub["trend_dev"] = (cost - roll_median_30).fillna(0.0)

    # Local trend slope: simple finite-difference of the 14-day rolling mean.
    sub["trend_slope"] = roll_mean_14.diff().fillna(0.0)

    # Lag ratios: how much higher than yesterday / a week ago is today?
    sub["lag_1_ratio"] = (cost / cost.shift(1).replace(0, np.nan)).fillna(1.0)
    sub["lag_7_ratio"] = (cost / cost.shift(7).replace(0, np.nan)).fillna(1.0)

    # Weekly-seasonality-aware residual: subtract the same-weekday median
    # over the trailing 4-week window. Spike survives, level shift survives,
    # benign weekly seasonality cancels out.
    dow_med = (
        cost.groupby(sub["date"].dt.weekday)
        .transform(lambda s: s.rolling(4, min_periods=1).median())
    )
    sub["season_resid"] = (cost - dow_med).fillna(0.0)

    dow = sub["date"].dt.weekday
    sub["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    sub["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Replace any inf produced by ratios with the column median.
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.bfill().ffill().fillna(0.0)
    return sub


def detect(
    long_df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
    n_estimators: int = 400,
    score_threshold: float = 0.55,
    group_keys: Sequence[str] = ("service",),
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, *``group_keys``, ``cost``.
        contamination: expected fraction of anomalous days per group.
            Tuned to the synthetic dataset's injected anomaly rate.
        n_estimators: forest size.
        score_threshold: secondary cutoff on the normalized [0, 1] score.
            A point must pass both ``predict()`` and exceed this to flag,
            which prunes the per-group over-flagging that contaminates
            groups with no real anomalies.
        group_keys: columns that identify an independent series. Defaults to
            ``("service",)`` for backward compatibility; pass
            ``("service", "env")`` for multi-granularity scoring — that is
            where IsolationForest's multivariate strength actually shows
            up versus structural decomposition (STL).
    """
    keys = list(group_keys)
    out = []
    for group_vals, sub in long_df.groupby(keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        feats = _features(sub)
        for k, v in zip(keys, group_vals):
            feats[k] = v
        X = feats[FEATURE_COLS].to_numpy()

        if len(X) < 14:
            feats["score"] = 0.0
            feats["is_anomaly"] = False
            out.append(feats[["date", *keys, "cost", "score", "is_anomaly"]])
            continue

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(X)

        raw = -model.decision_function(X)
        rng = raw.max() - raw.min()
        score = (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)
        predicted = model.predict(X) == -1
        feats["score"] = score
        feats["is_anomaly"] = predicted & (score >= score_threshold)
        out.append(feats[["date", *keys, "cost", "score", "is_anomaly"]])

    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["date", *keys])
        .reset_index(drop=True)
    )
