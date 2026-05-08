"""Detector runtime benchmarks — measured, not estimated.

The Cloud architecture diagram (REPORT § 4.1) prescribes ECS Fargate
for the detection pass; rightsizing that container needs an honest
"how long does each detector take?" number. This module measures it.

Used by the dashboard's Performance tab. Cheap to run (median of 3
calls per detector × dataset_size) so it can refresh on demand.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from .detectors import DETECTORS
from .preprocessing import aggregate_by_service
from .synthetic_data import generate


@dataclass
class PerfRow:
    detector: str
    n_days: int
    n_services: int
    seconds_per_run: float
    rows_processed: int
    rows_per_second: float


def time_detector(
    detector_name: str,
    long_df: pd.DataFrame,
    *,
    repeat: int = 3,
) -> PerfRow:
    """Time one detector on a long-format dataframe (median of `repeat`)."""
    fn = DETECTORS[detector_name]
    runs: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(long_df)
        runs.append(time.perf_counter() - t0)
    runs.sort()
    median_s = runs[len(runs) // 2]
    n_services = int(long_df["service"].nunique())
    n_days = int(long_df["date"].nunique())
    rows = len(long_df)
    return PerfRow(
        detector=detector_name,
        n_days=n_days,
        n_services=n_services,
        seconds_per_run=median_s,
        rows_processed=rows,
        rows_per_second=rows / max(median_s, 1e-9),
    )


def benchmark_grid(
    n_days_options: tuple[int, ...] = (30, 60, 90, 120),
    seed: int = 42,
) -> pd.DataFrame:
    """Run every detector across a grid of dataset sizes; return one row each."""
    rows: list[PerfRow] = []
    for n_days in n_days_options:
        cur, _, _ = generate(n_days=n_days, seed=seed)
        long = aggregate_by_service(cur)
        for det_name in DETECTORS.keys():
            rows.append(time_detector(det_name, long))

    return pd.DataFrame([
        {
            "detector": r.detector,
            "n_days": r.n_days,
            "n_services": r.n_services,
            "seconds_per_run": round(r.seconds_per_run, 4),
            "rows_processed": r.rows_processed,
            "rows_per_second": round(r.rows_per_second, 1),
        }
        for r in rows
    ])
