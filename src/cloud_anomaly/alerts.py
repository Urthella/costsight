"""Alert module — turns flagged points into ranked, severity-banded alerts.

Severity definition (matches the proposal slide):
    severity = deviation × duration × dollar_impact

    * deviation: detector score, normalized to [0, 1]
    * duration: number of consecutive anomaly days the point belongs to
                (also normalized vs. dataset length)
    * dollar_impact: cost on the day, normalized vs. service rolling mean

Final severity is mapped to LOW / MEDIUM / HIGH bands.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS_DIR, SEVERITY_BANDS


def _band(value: float) -> str:
    for name, (lo, hi) in SEVERITY_BANDS.items():
        if lo <= value < hi:
            return name
    return "HIGH"


def _consecutive_lengths(flags: pd.Series) -> pd.Series:
    """For each True, length of the contiguous True run it belongs to."""
    flags = flags.astype(bool).reset_index(drop=True)
    lengths = pd.Series(0, index=flags.index, dtype=int)
    run_id = (flags != flags.shift()).cumsum()
    for _, idx in flags.groupby(run_id).groups.items():
        if flags.iloc[idx[0]]:
            lengths.iloc[idx] = len(idx)
    return lengths


def build_alerts(
    detections: pd.DataFrame,
    detector_name: str,
    dataset_days: int | None = None,
) -> pd.DataFrame:
    """Convert per-day detector flags into an alert log.

    Args:
        detections: ``detect()`` output (date, service, cost, score, is_anomaly).
        detector_name: label written into the alert rows.
        dataset_days: total days in the source dataset (for duration norm).
    """
    df = detections.copy()
    if dataset_days is None:
        dataset_days = df["date"].nunique() or 1

    # Normalize score to [0, 1] per detector run.
    score = df["score"].astype(float)
    s_min, s_max = score.min(), score.max()
    deviation = (score - s_min) / (s_max - s_min) if s_max > s_min else score * 0

    # Duration: contiguous flagged-day run length per service.
    durations = []
    for service, sub in df.groupby("service"):
        durations.append(
            pd.DataFrame(
                {
                    "date": sub["date"].values,
                    "service": service,
                    "_duration": _consecutive_lengths(sub["is_anomaly"]).values,
                }
            )
        )
    dur_df = pd.concat(durations, ignore_index=True)
    df = df.merge(dur_df, on=["date", "service"], how="left")

    # Dollar impact: cost on day / service mean (cap at 5x).
    service_mean = df.groupby("service")["cost"].transform("mean").replace(0, np.nan)
    dollar_norm = (df["cost"] / service_mean).clip(upper=5.0).fillna(1.0) / 5.0

    duration_norm = (df["_duration"] / max(dataset_days, 1)).clip(upper=1.0)

    severity = (deviation.fillna(0.0) * (0.4 + 0.6 * duration_norm) * (0.4 + 0.6 * dollar_norm))
    severity = severity.clip(0.0, 1.0)

    df["severity_score"] = severity
    df["severity"] = severity.apply(_band)
    df["detector"] = detector_name

    alerts = (
        df.loc[df["is_anomaly"], [
            "date", "service", "cost", "score", "severity_score", "severity", "detector",
        ]]
        .sort_values(["severity_score", "date"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return alerts


def write_alerts(alerts: pd.DataFrame, detector_name: str, out_dir: Path | None = None) -> dict[str, Path]:
    """Persist alerts as both JSON and CSV; returns the written paths."""
    out_dir = out_dir or OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"alerts_{detector_name}.csv"
    json_path = out_dir / f"alerts_{detector_name}.json"

    payload = alerts.copy()
    payload["date"] = payload["date"].dt.strftime("%Y-%m-%d")
    payload.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload.to_dict(orient="records"), indent=2))

    return {"csv": csv_path, "json": json_path}
