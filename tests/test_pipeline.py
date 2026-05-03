"""Smoke tests for the end-to-end pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.alerts import build_alerts
from cloud_anomaly.benchmark import run as run_benchmark
from cloud_anomaly.config import RAW_DIR
from cloud_anomaly.detectors import DETECTORS
from cloud_anomaly.evaluation import compare_detectors, evaluate, evaluate_alerts
from cloud_anomaly.pipeline import detector_kwargs
from cloud_anomaly.preprocessing import aggregate_by, aggregate_by_service
from cloud_anomaly.synthetic_data import generate


def test_synthetic_dataset_shape():
    cur, labels, anomalies = generate(n_days=60, seed=7)
    assert {"date", "service", "region", "usage_type", "env", "cost"} <= set(cur.columns)
    assert (cur["cost"] > 0).all()
    assert labels["is_anomaly"].sum() > 0
    assert len(anomalies) >= 4
    # Each anomaly has an explicit env; anomalies span more than one env.
    assert {a.env for a in anomalies} - {"prod", "staging", "dev"} == set()
    assert len({a.env for a in anomalies}) >= 2

    # Service-level cost is preserved bit-for-bit when env is summed back.
    by_service = cur.groupby(["date", "service"], as_index=False)["cost"].sum()
    by_service_env = (
        cur.groupby(["date", "service", "env"], as_index=False)["cost"]
        .sum()
        .groupby(["date", "service"], as_index=False)["cost"]
        .sum()
    )
    merged = by_service.merge(by_service_env, on=["date", "service"], suffixes=("_a", "_b"))
    assert (merged["cost_a"] - merged["cost_b"]).abs().max() < 1e-6


def test_granular_labels_written_and_consistent():
    generate(n_days=60, seed=7)
    granular = pd.read_csv(
        RAW_DIR / "ground_truth_labels_granular.csv", parse_dates=["date"]
    )
    assert {"date", "service", "env", "is_anomaly", "anomaly_type"} <= set(granular.columns)
    # Every anomalous granular row must have an env that exists in CUR.
    cur, labels_svc, _ = generate(n_days=60, seed=7)
    assert set(granular["env"].unique()) == set(cur["env"].unique())
    # Service-level labels are the env-collapsed view of the granular table.
    rebuilt = (
        granular.sort_values("anomaly_type", ascending=False)
        .groupby(["date", "service"], as_index=False)
        .agg({"is_anomaly": "any"})
    )
    assert int(rebuilt["is_anomaly"].sum()) == int(labels_svc["is_anomaly"].sum())


def test_each_detector_runs():
    cur, _, _ = generate(n_days=60, seed=7)
    long = aggregate_by_service(cur)
    for name, fn in DETECTORS.items():
        det = fn(long)
        assert {"date", "service", "cost", "score", "is_anomaly"} <= set(det.columns)
        assert len(det) == len(long)


def test_alerts_and_eval():
    cur, labels, _ = generate(n_days=60, seed=7)
    long = aggregate_by_service(cur)
    detectors = {name: fn(long) for name, fn in DETECTORS.items()}

    for name, det in detectors.items():
        alerts = build_alerts(det, detector_name=name, dataset_days=60)
        if not alerts.empty:
            assert alerts["severity"].isin(["LOW", "MEDIUM", "HIGH"]).all()

    metrics = evaluate(detectors["zscore"], labels)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0

    comparison = compare_detectors(detectors, labels)
    assert {"detector", "anomaly_type", "precision", "recall", "f1"} <= set(comparison.columns)

    # Alert quality breakdown should report severity bands when alerts exist.
    alerts_stl = build_alerts(detectors["stl"], detector_name="stl", dataset_days=60)
    if not alerts_stl.empty:
        quality = evaluate_alerts(alerts_stl, labels)
        assert {"severity", "n_alerts", "true_positive", "precision"} <= set(quality.columns)
        assert quality["precision"].between(0.0, 1.0).all()


def test_benchmark_runs():
    result = run_benchmark(n_seeds=3, n_days=60, base_seed=2000)
    assert {"detector", "anomaly_type", "f1_mean", "f1_std", "n_runs"} <= set(result.summary.columns)
    assert (result.summary["n_runs"] == 3).all()
    assert result.raw["seed"].nunique() == 3


def test_multi_granularity_pipeline_runs():
    """End-to-end smoke test for the (service, env) granularity path."""
    generate(n_days=60, seed=11)
    granular = pd.read_csv(
        RAW_DIR / "ground_truth_labels_granular.csv", parse_dates=["date"]
    )
    cur, _, _ = generate(n_days=60, seed=11)
    long = aggregate_by(cur, ["service", "env"])
    n_groups = long.groupby(["service", "env"]).ngroups
    # 7 services × 3 envs.
    assert n_groups == 21

    detectors_out = {}
    for name, fn in DETECTORS.items():
        extra = detector_kwargs(name, ("service", "env"), n_groups)
        det = fn(long, group_keys=("service", "env"), **extra)
        assert {"date", "service", "env", "cost", "score", "is_anomaly"} <= set(det.columns)
        assert len(det) == len(long)
        detectors_out[name] = det

    # IForest gets a smaller contamination in multi-gran mode.
    assert detector_kwargs("iforest", ("service", "env"), 21)["contamination"] < 0.08
    assert detector_kwargs("iforest", ("service",), 7) == {}

    comparison = compare_detectors(detectors_out, granular, group_keys=("service", "env"))
    assert (comparison["precision"].between(0.0, 1.0)).all()
    assert (comparison["recall"].between(0.0, 1.0)).all()

    # Alerts carry the env column when granularity includes it.
    alerts = build_alerts(detectors_out["stl"], "stl", dataset_days=60, group_keys=["service", "env"])
    if not alerts.empty:
        assert "env" in alerts.columns
        assert alerts["env"].isin({"prod", "staging", "dev"}).all()


def test_benchmark_multi_granularity_runs():
    result = run_benchmark(
        n_seeds=2, n_days=60, base_seed=3000, group_keys=("service", "env")
    )
    assert {"detector", "anomaly_type", "f1_mean", "f1_std", "n_runs"} <= set(result.summary.columns)
    assert (result.summary["n_runs"] == 2).all()
