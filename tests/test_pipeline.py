"""Smoke tests for the end-to-end pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.alerts import build_alerts
from cloud_anomaly.attribution import attribute
from cloud_anomaly.benchmark import run as run_benchmark
from cloud_anomaly.detectors import DETECTORS
from cloud_anomaly.evaluation import (
    compare_detectors,
    cost_saved_estimate,
    evaluate,
    evaluate_alerts,
    time_to_detect,
)
from cloud_anomaly.preprocessing import aggregate_by_service
from cloud_anomaly.synthetic_data import generate


def test_synthetic_dataset_shape():
    cur, labels, anomalies = generate(n_days=60, seed=7)
    assert {"date", "service", "region", "usage_type", "cost"} <= set(cur.columns)
    assert (cur["cost"] > 0).all()
    assert labels["is_anomaly"].sum() > 0
    assert len(anomalies) >= 4


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


def test_attribution_runs():
    cur, _, _ = generate(n_days=60, seed=11)
    long = aggregate_by_service(cur)
    detections = DETECTORS["stl"](long)
    alerts = build_alerts(detections, detector_name="stl", dataset_days=60)

    if alerts.empty:
        return

    attribution = attribute(cur, alerts)
    assert {
        "date", "service", "total_cost", "baseline_cost", "delta",
        "severity", "top_dimension", "top_value", "top_value_share",
        "top_value_delta", "summary",
    } <= set(attribution.columns)
    assert (attribution["top_value_share"].between(0.0, 1.0)).all()
    # Empty alerts → empty (but still well-shaped) attribution frame.
    empty = attribute(cur, alerts.iloc[:0])
    assert empty.empty


def test_ensemble_detector():
    cur, _, _ = generate(n_days=60, seed=11)
    long = aggregate_by_service(cur)
    ensemble = DETECTORS["ensemble"](long)
    assert {"date", "service", "cost", "score", "is_anomaly"} <= set(ensemble.columns)
    assert len(ensemble) == len(long)


def test_time_to_detect_and_cost_saved():
    cur, labels, _ = generate(n_days=60, seed=11)
    long = aggregate_by_service(cur)
    detections = DETECTORS["stl"](long)

    ttd = time_to_detect(detections, labels)
    assert {
        "service", "anomaly_type", "window_start", "window_end", "days_to_detect",
    } <= set(ttd.columns)
    finite = ttd["days_to_detect"].dropna()
    if not finite.empty:
        assert (finite >= 0).all()

    saved = cost_saved_estimate(cur, detections, labels)
    assert set(saved.keys()) == {"saved", "total_anomaly_cost", "ratio"}
    assert saved["saved"] >= 0
    assert saved["total_anomaly_cost"] >= 0
    assert 0 <= saved["ratio"] <= 1.0
