"""Robust Z-Score + CUSUM baseline detector.

Two complementary, upward-biased signals run independently per service on the
daily long-format frame. Cost anomalies are *overspends*, so both signals look
for upward deviations only - flagging the return-to-normal after a spike as a
"downward anomaly" only manufactures false positives.

* **Robust point z-score** over a *trailing* window (median / MAD). The window
  excludes the current day, so an anomalous point can no longer inflate its own
  baseline and mask itself; MAD keeps the baseline from being dragged by a
  recent spike. A relative floor on the scale stops a near-flat window from
  turning ordinary jitter into a huge z. This catches point spikes and the
  onset of a level shift.
* **One-sided CUSUM** on the standardised residuals. A point z-score is blind
  to sustained change - a level shift or slow drift gets absorbed into the
  moving baseline within a window - so CUSUM accumulates small persistent
  positive deviations and fires once the evidence crosses a decision limit.

A point is flagged if either signal trips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_TO_SIGMA = 1.4826  # scales MAD to a std-equivalent for Gaussian data


def _robust_residuals(
    cost: np.ndarray, window: int, min_periods: int, rel_floor: float
) -> np.ndarray:
    """Standardised deviation of each day from its trailing robust baseline.

    Returns an array of z-like residuals (0 where the baseline is undefined).
    The scale is floored at ``rel_floor * |median|`` so a quiet, near-constant
    window cannot explode ordinary fluctuations into spurious anomalies.
    """
    n = cost.size
    resid = np.zeros(n)
    for t in range(n):
        base = cost[max(0, t - window):t]  # strictly before t -> no self-masking
        if base.size < min_periods:
            continue
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med)))
        scale = _MAD_TO_SIGMA * mad
        if scale <= 1e-9:  # flat window -> MAD is 0; fall back to std
            std = float(base.std(ddof=0))
            scale = std if std > 1e-9 else np.nan
        if np.isfinite(scale):
            scale = max(scale, rel_floor * abs(med))
            if scale > 1e-9:
                resid[t] = (cost[t] - med) / scale
    return resid


def _cusum_upper(resid: np.ndarray, slack: float) -> np.ndarray:
    """One-sided (upward) tabular CUSUM magnitude per point."""
    s_hi = 0.0
    out = np.zeros(resid.size)
    for t in range(resid.size):
        s_hi = max(0.0, s_hi + resid[t] - slack)
        out[t] = s_hi
    return out


def detect(
    long_df: pd.DataFrame,
    window: int = 14,
    threshold: float = 3.5,
    cusum_slack: float = 1.0,
    cusum_threshold: float = 5.0,
    rel_floor: float = 0.10,
) -> pd.DataFrame:
    """Args:
        long_df: columns ``date``, ``service``, ``cost``.
        window: trailing rolling window size in days.
        threshold: robust z above this is flagged (point spikes).
        cusum_slack: per-step allowance ``k`` before CUSUM accumulates.
        cusum_threshold: CUSUM decision limit ``h`` (sustained shift / drift).
        rel_floor: scale floor as a fraction of the baseline median.
    """
    if long_df.empty:
        return pd.DataFrame(
            columns=["date", "service", "cost", "score", "is_anomaly"]
        ).astype({"score": "float64", "is_anomaly": "bool"})

    min_periods = max(window // 2, 3)
    out = []
    for service, sub in long_df.groupby("service"):
        sub = sub.sort_values("date").copy()
        cost = sub["cost"].to_numpy(dtype=float)

        resid = _robust_residuals(cost, window, min_periods, rel_floor)
        cusum = _cusum_upper(resid, cusum_slack)

        point_hit = resid >= threshold
        cusum_hit = cusum >= cusum_threshold

        # Put both signals on the same ~sigma scale: a CUSUM at its threshold
        # reads as `threshold`, so downstream severity banding stays meaningful.
        cusum_scaled = cusum / cusum_threshold * threshold
        sub["score"] = np.maximum(np.clip(resid, 0.0, None), cusum_scaled)
        sub["is_anomaly"] = point_hit | cusum_hit
        sub["service"] = service
        out.append(sub[["date", "service", "cost", "score", "is_anomaly"]])

    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["date", "service"])
        .reset_index(drop=True)
    )
