"""Isolation Forest detector.

Trains one IsolationForest per service on a multivariate feature set
designed to expose all three target anomaly types:

    - cost              raw daily spend
    - log_cost          log scale (heavy-tail robustness)
    - rel_to_median     cost / 14-day rolling median (point spikes pop)
    - rel_to_mean       cost / 14-day rolling mean
    - roll_std_14       rolling std (level-shift signature)
    - pct_change        day-over-day change (sudden moves)
    - trend_dev         cost minus 28-day rolling median (drift signature)
    - dow_sin / dow_cos cyclic day-of-week encoding

Anomalies are flagged via the model's native ``predict`` (driven by
``contamination``); the per-service min-max-normalized
``-decision_function`` is exposed as ``score``.
"""
from __future__ import annotations

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
    "dow_sin",
    "dow_cos",
]


def _features(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("date").copy()
    cost = sub["cost"].astype(float)

    roll_median_14 = cost.rolling(14, min_periods=3).median()
    roll_mean_14 = cost.rolling(14, min_periods=3).mean()
    roll_median_28 = cost.rolling(28, min_periods=3).median()

    sub["log_cost"] = np.log1p(cost)
    sub["rel_to_median"] = (cost / roll_median_14.replace(0, np.nan)).fillna(1.0)
    sub["rel_to_mean"] = (cost / roll_mean_14.replace(0, np.nan)).fillna(1.0)
    sub["roll_std_14"] = cost.rolling(14, min_periods=3).std().fillna(0.0)
    sub["pct_change"] = cost.pct_change().fillna(0.0)
    sub["trend_dev"] = (cost - roll_median_28).fillna(0.0)

    dow = sub["date"].dt.weekday
    sub["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    sub["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return sub.bfill().ffill()


def detect(
    long_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 300,
    score_threshold: float = 0.6,
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, ``service``, ``cost``.
        contamination: expected fraction of anomalous days per service —
            kept low so sklearn's intrinsic cutoff is conservative.
        n_estimators: forest size.
        score_threshold: secondary cutoff on the normalized [0, 1] score.
            A point must pass both ``predict()`` and exceed this to flag,
            which prunes noisy service-level over-flagging.
    """
    out = []
    for service, sub in long_df.groupby("service"):
        feats = _features(sub)
        X = feats[FEATURE_COLS].to_numpy()

        if len(X) < 14:
            feats["score"] = 0.0
            feats["is_anomaly"] = False
            out.append(feats[["date", "service", "cost", "score", "is_anomaly"]])
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
        feats["service"] = service
        out.append(feats[["date", "service", "cost", "score", "is_anomaly"]])

    return pd.concat(out, ignore_index=True).sort_values(["date", "service"]).reset_index(drop=True)
