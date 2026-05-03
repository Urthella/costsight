"""Generate a synthetic AWS Cost & Usage Report (CUR) dataset.

The output mimics the daily-granularity CUR schema: one row per
(date, service, region, usage_type) with a positive ``cost`` value.

Three anomaly types are injected with deterministic ground-truth labels
so the same dataset can be used to compute Precision/Recall later:

  * point_spike   — single-day cost explosion (e.g. infinite loop)
  * level_shift   — persistent step up after some change
  * gradual_drift — slow upward creep over a window
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DAYS, DEFAULT_SEED, RAW_DIR, SERVICES


@dataclass(frozen=True)
class InjectedAnomaly:
    """One injected anomaly — used as ground truth for evaluation."""

    service: str
    anomaly_type: str
    start_day: int
    end_day: int      # inclusive
    multiplier: float

    def affects(self, day: int, service: str) -> bool:
        return service == self.service and self.start_day <= day <= self.end_day


def _build_baseline(rng: np.random.Generator, n_days: int) -> pd.DataFrame:
    """Per-(day, service) clean cost with weekly seasonality + small noise."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for service, region, usage_type, base_cost, noise_pct in SERVICES:
        # Weekly seasonality: weekdays slightly heavier than weekends.
        dow = np.array([d.weekday() for d in dates])
        weekly = 1.0 + 0.12 * np.where(dow < 5, 1.0, -0.6)
        # Mild monthly drift (1% over 30 days).
        monthly = 1.0 + 0.01 * (np.arange(n_days) / 30.0)
        noise = rng.normal(loc=1.0, scale=noise_pct, size=n_days).clip(0.4, 1.8)
        cost = base_cost * weekly * monthly * noise
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "service": service,
                    "region": region,
                    "usage_type": usage_type,
                    "cost": cost,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _default_anomalies(n_days: int, rng: np.random.Generator) -> list[InjectedAnomaly]:
    """Pick a representative set of anomalies, scaled to ``n_days``.

    All day indices are in [0, n_days - 1] so the generator works for
    arbitrary window sizes the dashboard slider exposes (30–180).
    """
    services = [s[0] for s in SERVICES]
    anomalies: list[InjectedAnomaly] = []
    last = n_days - 1

    spike_ec2 = max(0, last - 12)
    spike_lambda = min(last, max(5, n_days // 4))
    level_start = min(last, max(7, int(n_days * 0.4)))
    level_end = min(last, level_start + max(7, n_days // 5))
    drift_start = min(last, max(level_end + 5, int(n_days * 0.65)))

    anomalies.append(InjectedAnomaly("EC2", "point_spike", spike_ec2, spike_ec2, 4.5))
    anomalies.append(InjectedAnomaly("Lambda", "point_spike", spike_lambda, spike_lambda, 6.0))
    anomalies.append(InjectedAnomaly("RDS", "level_shift", level_start, level_end, 1.7))
    if drift_start < last:
        anomalies.append(InjectedAnomaly("S3", "gradual_drift", drift_start, last, 2.0))

    # A randomly-timed extra spike on a smaller service for variety per seed.
    extras = [s for s in services if s not in {"EC2", "S3", "RDS", "Lambda"}]
    if extras and n_days > 15:
        extra_service = rng.choice(extras)
        extra_day = int(rng.integers(low=5, high=max(6, n_days - 5)))
        anomalies.append(
            InjectedAnomaly(str(extra_service), "point_spike", extra_day, extra_day, 3.5)
        )

    return anomalies


def _apply_anomalies(
    df: pd.DataFrame, anomalies: list[InjectedAnomaly], n_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multiply costs in place and return (rows, ground_truth_label_table)."""
    df = df.copy()
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    for anom in anomalies:
        mask = (df["service"] == anom.service) & df["day_index"].between(
            anom.start_day, anom.end_day
        )
        if anom.anomaly_type == "gradual_drift":
            # Linear ramp from 1.0 → multiplier across the window.
            length = max(anom.end_day - anom.start_day, 1)
            ramp = 1.0 + (anom.multiplier - 1.0) * (
                (df.loc[mask, "day_index"] - anom.start_day) / length
            )
            df.loc[mask, "cost"] *= ramp
        else:
            df.loc[mask, "cost"] *= anom.multiplier

    # Build the ground-truth label table: one row per (date, service).
    dates = pd.date_range(df["date"].min(), periods=n_days, freq="D")
    services = sorted({a.service for a in anomalies} | set(df["service"].unique()))
    labels = pd.DataFrame(
        [(d, s) for d in dates for s in services], columns=["date", "service"]
    )
    labels["is_anomaly"] = False
    labels["anomaly_type"] = ""

    for anom in anomalies:
        day_mask = labels["date"].between(
            dates[anom.start_day], dates[anom.end_day]
        )
        svc_mask = labels["service"] == anom.service
        labels.loc[day_mask & svc_mask, "is_anomaly"] = True
        labels.loc[day_mask & svc_mask, "anomaly_type"] = anom.anomaly_type

    return df.drop(columns=["day_index"]), labels


def generate(
    n_days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[InjectedAnomaly]]:
    """Generate a synthetic CUR dataset and write CSV + Parquet to ``out_dir``.

    Returns ``(cur_df, labels_df, anomalies)``.
    """
    rng = np.random.default_rng(seed)
    baseline = _build_baseline(rng, n_days)
    anomalies = _default_anomalies(n_days, rng)
    cur_df, labels_df = _apply_anomalies(baseline, anomalies, n_days)

    cur_df = cur_df.sort_values(["date", "service"]).reset_index(drop=True)
    cur_df["cost"] = cur_df["cost"].round(2)

    out_dir = out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cur_df.to_csv(out_dir / "cur_synthetic.csv", index=False)
    cur_df.to_parquet(out_dir / "cur_synthetic.parquet", index=False)
    labels_df.to_csv(out_dir / "ground_truth_labels.csv", index=False)

    return cur_df, labels_df, anomalies


if __name__ == "__main__":
    cur_df, labels_df, anomalies = generate()
    print(f"Wrote {len(cur_df)} CUR rows across {cur_df['service'].nunique()} services.")
    print(f"Injected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"  - {a.service:<10} {a.anomaly_type:<14} days {a.start_day}-{a.end_day}  ×{a.multiplier}")
    print(f"Ground-truth positive rows: {labels_df['is_anomaly'].sum()}")
