"""Concept drift detection — when the *normal* itself shifts.

The anomaly detectors all assume the baseline is stable: they flag
points that deviate from a rolling mean. But what if the baseline is
*itself* drifting? Then detector thresholds need to be recalibrated,
or the rolling window needs to be shorter.

This module implements two classical drift detectors operating on the
per-service daily cost stream:

  * **Page-Hinkley test** — a one-pass online change-point detector.
    Tracks the cumulative deviation from a running mean, flags when
    the deviation exceeds ``threshold * sigma``.
  * **ADWIN-lite** — Adaptive Windowing. Maintains a sliding window
    and asks "has the mean of the recent half-window diverged from
    the older half?". When yes, the older half is dropped.

Output is a list of drift events with (service, change_date,
direction, magnitude). The dashboard renders these as gold dashed
lines on the cost-trend chart so reviewers see *baseline shifts*
distinctly from *anomaly spikes*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftEvent:
    service: str
    change_date: pd.Timestamp
    direction: str        # "up" or "down"
    magnitude_pct: float  # % change from old mean to new mean
    detector: str         # "page-hinkley" or "adwin"
    confidence: float     # 0-1


def page_hinkley(
    series: pd.Series,
    *,
    delta: float = 0.005,
    threshold: float = 50.0,
) -> list[int]:
    """Return the indices in `series` where Page-Hinkley flags a change.

    delta is the allowed magnitude of changes (slack), threshold is the
    cumulative-deviation level above which we declare a change.
    """
    if len(series) < 10:
        return []
    values = series.to_numpy(dtype=float)
    mean = values[0]
    cum_sum = 0.0
    min_sum = 0.0
    flags: list[int] = []
    for i in range(1, len(values)):
        mean = mean + (values[i] - mean) / (i + 1)
        cum_sum += values[i] - mean - delta
        min_sum = min(min_sum, cum_sum)
        if (cum_sum - min_sum) > threshold:
            flags.append(i)
            # Reset state so we can find subsequent change-points.
            mean = values[i]
            cum_sum = 0.0
            min_sum = 0.0
    return flags


def adwin_lite(
    series: pd.Series,
    *,
    min_window: int = 7,
    sensitivity: float = 0.30,
) -> list[int]:
    """Return change-point indices via a lightweight Adaptive Windowing scheme.

    Splits the rolling window at every point and asks: has the recent
    half's mean diverged from the older half's mean by more than
    ``sensitivity`` (relative)? The first index where it does → drift.
    """
    if len(series) < 2 * min_window:
        return []
    values = series.to_numpy(dtype=float)
    flags: list[int] = []
    i = 2 * min_window
    last_flag = -1
    while i < len(values):
        old = values[max(last_flag + 1, i - 2 * min_window):i - min_window]
        new = values[i - min_window:i]
        if len(old) < min_window or len(new) < min_window:
            i += 1
            continue
        old_mean = float(np.mean(old))
        new_mean = float(np.mean(new))
        if old_mean > 0 and abs(new_mean - old_mean) / old_mean > sensitivity:
            flags.append(i)
            last_flag = i
            i += min_window  # skip past the change so we don't double-flag
        else:
            i += 1
    return flags


def detect_drift(long_df: pd.DataFrame) -> pd.DataFrame:
    """Run both detectors per service; return a unified DriftEvent table.

    A drift event is recorded for any (service, date) flagged by either
    detector. Magnitude is computed against the per-service mean of the
    14 days preceding the change.
    """
    if long_df.empty:
        return pd.DataFrame(columns=[
            "service", "change_date", "direction", "magnitude_pct",
            "detector", "confidence",
        ])

    rows: list[DriftEvent] = []
    for service, group in long_df.groupby("service"):
        group = group.sort_values("date").reset_index(drop=True)
        costs = group["cost"]

        for idx in page_hinkley(costs):
            event = _build_event(group, idx, "page-hinkley", confidence=0.75)
            if event is not None:
                rows.append(event)

        for idx in adwin_lite(costs):
            event = _build_event(group, idx, "adwin", confidence=0.65)
            if event is not None:
                rows.append(event)

    if not rows:
        return pd.DataFrame(columns=[
            "service", "change_date", "direction", "magnitude_pct",
            "detector", "confidence",
        ])

    df = pd.DataFrame([
        {
            "service": r.service,
            "change_date": r.change_date,
            "direction": r.direction,
            "magnitude_pct": round(r.magnitude_pct, 1),
            "detector": r.detector,
            "confidence": r.confidence,
        }
        for r in rows
    ])
    return df.sort_values(["change_date", "service"]).reset_index(drop=True)


def _build_event(
    group: pd.DataFrame,
    idx: int,
    detector: str,
    *,
    confidence: float,
) -> DriftEvent | None:
    """Compute direction + magnitude for a change-point at index idx."""
    if idx <= 7 or idx >= len(group):
        return None
    before = group.iloc[max(0, idx - 14):idx]["cost"].mean()
    after = group.iloc[idx:idx + 14]["cost"].mean() if idx + 14 < len(group) else group.iloc[idx:]["cost"].mean()
    if not before or pd.isna(before):
        return None
    magnitude = (after - before) / before * 100
    return DriftEvent(
        service=str(group["service"].iloc[0]),
        change_date=pd.Timestamp(group["date"].iloc[idx]),
        direction="up" if magnitude > 0 else "down",
        magnitude_pct=float(abs(magnitude)),
        detector=detector,
        confidence=confidence,
    )
