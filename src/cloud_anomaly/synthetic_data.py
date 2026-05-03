"""Generate a synthetic AWS Cost & Usage Report (CUR) dataset.

The output mimics the daily-granularity CUR schema: one row per
(date, service, region, usage_type, env) with a positive ``cost`` value.
Each service is split across prod/staging/dev environments with weights
that sum to 1.0, so summing ``env`` reproduces the legacy single-env
baseline exactly — service-level aggregation stays unchanged.

Three anomaly types are injected with deterministic ground-truth labels
so the same dataset can be used to compute Precision/Recall later:

  * point_spike   — single-day cost explosion (e.g. infinite loop)
  * level_shift   — persistent step up after some change
  * gradual_drift — slow upward creep over a window

Each anomaly targets a specific (service, env) pair. The ground-truth
label table carries both keys so multi-granularity detectors can be
scored fairly while service-level evaluation still works (env-agnostic
match).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DAYS, DEFAULT_SEED, ENVIRONMENTS, RAW_DIR, SERVICES


@dataclass(frozen=True)
class InjectedAnomaly:
    """One injected anomaly — used as ground truth for evaluation."""

    service: str
    anomaly_type: str
    start_day: int
    end_day: int      # inclusive
    multiplier: float
    env: str = "prod"

    def affects(self, day: int, service: str, env: str | None = None) -> bool:
        ok = service == self.service and self.start_day <= day <= self.end_day
        if env is not None:
            ok = ok and env == self.env
        return ok


def _build_baseline(rng: np.random.Generator, n_days: int) -> pd.DataFrame:
    """Per-(day, service, env) clean cost with weekly seasonality + small noise.

    Each service's daily cost is split across the configured environments
    according to ENVIRONMENTS shares; weights sum to 1.0 so summing env
    reproduces the legacy single-env baseline exactly.
    """
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for service, region, usage_type, base_cost, noise_pct in SERVICES:
        # Weekly seasonality: weekdays slightly heavier than weekends.
        dow = np.array([d.weekday() for d in dates])
        weekly = 1.0 + 0.12 * np.where(dow < 5, 1.0, -0.6)
        # Mild monthly drift (1% over 30 days).
        monthly = 1.0 + 0.01 * (np.arange(n_days) / 30.0)
        for env, share, noise_mult in ENVIRONMENTS:
            env_noise_pct = noise_pct * noise_mult
            noise = rng.normal(loc=1.0, scale=env_noise_pct, size=n_days).clip(0.4, 1.8)
            cost = base_cost * share * weekly * monthly * noise
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "service": service,
                        "region": region,
                        "usage_type": usage_type,
                        "env": env,
                        "cost": cost,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _default_anomalies(n_days: int, rng: np.random.Generator) -> list[InjectedAnomaly]:
    """Pick a representative set of anomalies, scaled to ``n_days``.

    All day indices are in [0, n_days - 1] so the generator works for
    arbitrary window sizes the dashboard slider exposes (30–180).

    Anomalies are spread across environments on purpose: a dev-only
    runaway loop, a staging level-shift, and a prod gradual drift create
    a setting where service-level aggregation hides signal a
    multi-granularity detector can recover.
    """
    services = [s[0] for s in SERVICES]
    anomalies: list[InjectedAnomaly] = []
    last = n_days - 1

    spike_ec2 = max(0, last - 12)
    spike_lambda = min(last, max(5, n_days // 4))
    level_start = min(last, max(7, int(n_days * 0.4)))
    level_end = min(last, level_start + max(7, n_days // 5))
    drift_start = min(last, max(level_end + 5, int(n_days * 0.65)))

    # EC2 prod spike — large absolute impact, easy to catch service-level.
    anomalies.append(InjectedAnomaly("EC2", "point_spike", spike_ec2, spike_ec2, 4.5, env="prod"))
    # Lambda dev spike — small absolute $$ but huge relative jump in dev;
    # service-level aggregation often dilutes it below threshold.
    anomalies.append(InjectedAnomaly("Lambda", "point_spike", spike_lambda, spike_lambda, 6.0, env="dev"))
    # RDS staging level shift — staging is the "right" env to catch
    # mis-sized instance redeploys before they hit prod.
    anomalies.append(InjectedAnomaly("RDS", "level_shift", level_start, level_end, 1.7, env="staging"))
    # S3 prod gradual drift — slow log accumulation in production.
    if drift_start < last:
        anomalies.append(InjectedAnomaly("S3", "gradual_drift", drift_start, last, 2.0, env="prod"))

    # A randomly-timed extra spike on a smaller service / random env for
    # per-seed variety.
    extras = [s for s in services if s not in {"EC2", "S3", "RDS", "Lambda"}]
    env_names = [e[0] for e in ENVIRONMENTS]
    if extras and n_days > 15:
        extra_service = rng.choice(extras)
        extra_env = rng.choice(env_names)
        extra_day = int(rng.integers(low=5, high=max(6, n_days - 5)))
        anomalies.append(
            InjectedAnomaly(
                str(extra_service), "point_spike", extra_day, extra_day, 3.5, env=str(extra_env)
            )
        )

    return anomalies


def _apply_anomalies(
    df: pd.DataFrame, anomalies: list[InjectedAnomaly], n_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multiply costs in place and return (rows, ground_truth_label_table).

    The label table carries one row per (date, service, env) so
    multi-granularity detectors can be scored at the same granularity at
    which anomalies were actually injected. Service-level evaluation
    keeps working: any (date, service) with at least one True env row
    is considered anomalous after env-agnostic aggregation.
    """
    df = df.copy()
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    for anom in anomalies:
        mask = (
            (df["service"] == anom.service)
            & (df["env"] == anom.env)
            & df["day_index"].between(anom.start_day, anom.end_day)
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

    # Build the (date, service, env) ground-truth label table.
    dates = pd.date_range(df["date"].min(), periods=n_days, freq="D")
    services = sorted({a.service for a in anomalies} | set(df["service"].unique()))
    envs = sorted(df["env"].unique())
    labels = pd.DataFrame(
        [(d, s, e) for d in dates for s in services for e in envs],
        columns=["date", "service", "env"],
    )
    labels["is_anomaly"] = False
    labels["anomaly_type"] = ""

    for anom in anomalies:
        day_mask = labels["date"].between(dates[anom.start_day], dates[anom.end_day])
        target = day_mask & (labels["service"] == anom.service) & (labels["env"] == anom.env)
        labels.loc[target, "is_anomaly"] = True
        labels.loc[target, "anomaly_type"] = anom.anomaly_type

    return df.drop(columns=["day_index"]), labels


def _collapse_to_service(granular: pd.DataFrame) -> pd.DataFrame:
    """Roll a (date, service, env) label table up to (date, service).

    A (date, service) is anomalous if *any* env row for it is anomalous.
    The first non-empty anomaly_type wins — sufficient for type-stratified
    metrics because injected anomalies don't share a (service, day) cell.
    """
    grouped = granular.sort_values("anomaly_type", ascending=False).groupby(
        ["date", "service"], as_index=False
    ).agg({"is_anomaly": "any", "anomaly_type": "first"})
    return grouped


def generate(
    n_days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[InjectedAnomaly]]:
    """Generate a synthetic CUR dataset and write CSV + Parquet to ``out_dir``.

    Returns ``(cur_df, labels_df, anomalies)`` where ``labels_df`` is the
    legacy service-level ground-truth (for backward-compatible callers).
    The env-granular table is also written to disk as
    ``ground_truth_labels_granular.csv`` for multi-granularity scoring.
    """
    rng = np.random.default_rng(seed)
    baseline = _build_baseline(rng, n_days)
    anomalies = _default_anomalies(n_days, rng)
    cur_df, labels_granular = _apply_anomalies(baseline, anomalies, n_days)

    cur_df = cur_df.sort_values(["date", "service", "env"]).reset_index(drop=True)
    cur_df["cost"] = cur_df["cost"].round(2)

    labels_service = _collapse_to_service(labels_granular)

    out_dir = out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cur_df.to_csv(out_dir / "cur_synthetic.csv", index=False)
    cur_df.to_parquet(out_dir / "cur_synthetic.parquet", index=False)
    labels_service.to_csv(out_dir / "ground_truth_labels.csv", index=False)
    labels_granular.to_csv(out_dir / "ground_truth_labels_granular.csv", index=False)

    return cur_df, labels_service, anomalies


if __name__ == "__main__":
    cur_df, labels_df, anomalies = generate()
    print(
        f"Wrote {len(cur_df)} CUR rows across "
        f"{cur_df['service'].nunique()} services × {cur_df['env'].nunique()} envs."
    )
    print(f"Injected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(
            f"  - {a.service:<10} {a.env:<8} {a.anomaly_type:<14} "
            f"days {a.start_day}-{a.end_day}  ×{a.multiplier}"
        )
    print(f"Service-level ground-truth positive rows: {labels_df['is_anomaly'].sum()}")
