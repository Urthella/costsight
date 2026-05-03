"""Multi-seed Monte Carlo benchmark.

Runs the full pipeline across multiple random seeds and reports the
mean and standard deviation of Precision, Recall and F1 for every
(detector, anomaly_type) pair. This converts the single-run point
estimates in ``examples/comparison.csv`` into statistically defensible
numbers for the Phase 2 report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS_DIR
from .detectors import DETECTORS
from .evaluation import compare_detectors
from .preprocessing import aggregate_by_service
from .synthetic_data import generate


@dataclass
class BenchmarkResult:
    summary: pd.DataFrame   # mean ± std across seeds
    raw: pd.DataFrame       # one row per (seed, detector, anomaly_type)


def run(
    n_seeds: int = 25,
    base_seed: int = 1000,
    n_days: int = 90,
) -> BenchmarkResult:
    """Repeat the full pipeline ``n_seeds`` times with different seeds."""
    rows: list[pd.DataFrame] = []
    for i in range(n_seeds):
        seed = base_seed + i
        cur, labels, _ = generate(n_days=n_days, seed=seed)
        long = aggregate_by_service(cur)
        detector_outputs = {name: fn(long) for name, fn in DETECTORS.items()}
        comp = compare_detectors(detector_outputs, labels)
        comp["seed"] = seed
        rows.append(comp)

    raw = pd.concat(rows, ignore_index=True)

    grouped = raw.groupby(["detector", "anomaly_type"], as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n_runs=("seed", "count"),
    )
    for col in [
        "precision_mean", "precision_std", "recall_mean",
        "recall_std", "f1_mean", "f1_std",
    ]:
        grouped[col] = grouped[col].round(4)
    return BenchmarkResult(summary=grouped, raw=raw)


def write(result: BenchmarkResult, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "benchmark_summary.csv"
    raw_path = out_dir / "benchmark_raw.csv"
    result.summary.to_csv(summary_path, index=False)
    result.raw.to_csv(raw_path, index=False)
    return {"summary": summary_path, "raw": raw_path}


def render_table(summary: pd.DataFrame) -> str:
    """Pretty Markdown table: detector + anomaly_type + 'mean ± std' per metric."""
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    rows = ["| Detector | Anomaly type | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|"]
    detector_label = {"zscore": "Z-Score", "stl": "STL", "iforest": "Isolation Forest"}
    type_order = ["point_spike", "level_shift", "gradual_drift", "OVERALL"]
    for det in ["zscore", "stl", "iforest"]:
        for atype in type_order:
            row = summary[(summary["detector"] == det) & (summary["anomaly_type"] == atype)]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                f"| {detector_label[det]} | {atype.replace('_', ' ').title()} "
                f"| {fmt(r['precision_mean'], r['precision_std'])} "
                f"| {fmt(r['recall_mean'], r['recall_std'])} "
                f"| **{fmt(r['f1_mean'], r['f1_std'])}** |"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    print("Running 25-seed Monte Carlo benchmark...")
    result = run(n_seeds=25)
    paths = write(result)
    print(f"\nWrote {paths['summary']}")
    print(f"Wrote {paths['raw']}\n")
    print(render_table(result.summary))
