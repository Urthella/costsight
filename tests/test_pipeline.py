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
    bootstrap_f1_ci,
    compare_detectors,
    cost_saved_estimate,
    evaluate,
    evaluate_alerts,
    paired_significance,
    time_to_detect,
)
from cloud_anomaly.forecast import forecast_per_service, projected_monthly_spend
from cloud_anomaly.preprocessing import aggregate_by_service
from cloud_anomaly.synthetic_data import generate


def test_synthetic_dataset_shape():
    cur, labels, anomalies = generate(n_days=60, seed=7)
    assert {
        "date", "service", "region", "usage_type", "cost",
        "tag_team", "tag_environment",
    } <= set(cur.columns)
    assert (cur["cost"] > 0).all()
    assert labels["is_anomaly"].sum() > 0
    assert len(anomalies) >= 4
    assert (cur["tag_team"] != "").all()
    assert (cur["tag_environment"] != "").all()


def test_scenario_presets():
    from cloud_anomaly.synthetic_data import SCENARIOS

    assert {"default", "drift_heavy", "spike_storm", "stealth_leak",
            "multi_region", "weekend_camouflage", "calm"} <= set(SCENARIOS.keys())

    # calm scenario should produce zero anomalies
    cur, labels, anomalies = generate(n_days=60, seed=11, scenario="calm")
    assert len(anomalies) == 0
    assert int(labels["is_anomaly"].sum()) == 0
    assert (cur["cost"] > 0).all()

    # drift_heavy should have only gradual_drift entries
    _, _, drift_anoms = generate(n_days=90, seed=13, scenario="drift_heavy")
    assert len(drift_anoms) >= 2
    assert all(a.anomaly_type == "gradual_drift" for a in drift_anoms)


def test_playbook_and_pricing():
    from cloud_anomaly.playbook import PLAYBOOKS, get
    from cloud_anomaly.pricing import lookup, estimated_monthly

    for atype in ["point_spike", "level_shift", "gradual_drift", "multi_detector_consensus"]:
        book = get(atype)
        assert book["headline"]
        assert book["owner"]
        assert book["sla"]

    fallback = get("nonexistent_type")
    assert "Unknown" in fallback["headline"]
    assert PLAYBOOKS  # at least one entry

    quotes = lookup("EC2", "us-east-1")
    assert len(quotes) >= 2
    assert all(q.unit_price > 0 for q in quotes)

    monthly = estimated_monthly("EC2", "us-east-1", "t3.medium")
    assert monthly > 0
    assert monthly < 200  # t3.medium can't be more than ~$30/mo, sanity bound


def test_notification_payload():
    from cloud_anomaly.notification import build_payload_from_alert

    row = {
        "date": pd.Timestamp("2025-04-03"),
        "service": "EC2",
        "cost": 957.00,
        "severity": "HIGH",
        "summary": "EC2 spend spiked 4×",
        "flagged_by": "stl, iforest",
    }
    payload = build_payload_from_alert(row, detector="stl")
    assert payload.severity == "HIGH"
    assert "EC2" in payload.title
    block = payload.to_slack_block()
    assert "blocks" in block
    sns = payload.to_sns()
    assert sns["Subject"].startswith("[HIGH]")


def test_real_cur_loader():
    from cloud_anomaly.cur_loader import load_cur_csv

    sample = ROOT / "examples" / "aws_cur_sample.csv"
    long = load_cur_csv(sample)
    assert {
        "date", "service", "region", "usage_type", "cost",
        "tag_team", "tag_environment",
    } <= set(long.columns)
    assert long["service"].str.contains("EC2").any()
    assert (long["cost"] >= 0).all()
    # The huge spike on 2025-04-03 EC2 should be captured.
    apr3_ec2 = long[
        (long["service"] == "EC2") & (long["date"] == pd.Timestamp("2025-04-03"))
    ]["cost"].sum()
    assert apr3_ec2 > 900


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


def test_bootstrap_and_significance():
    from cloud_anomaly.benchmark import run as run_benchmark
    result = run_benchmark(n_seeds=5, n_days=60, base_seed=3000)
    raw = result.raw

    ci = bootstrap_f1_ci(raw, "stl", "OVERALL", n_resamples=200)
    assert {"mean", "lo", "hi", "n"} <= set(ci.keys())
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["n"] == 5

    sig = paired_significance(raw, "stl", "zscore", anomaly_type="OVERALL")
    assert {"statistic", "p_value", "n"} <= set(sig.keys())
    assert sig["n"] == 5
    if not pd.isna(sig["p_value"]):
        assert 0.0 <= sig["p_value"] <= 1.0


def test_api_smoke():
    from fastapi.testclient import TestClient
    from cloud_anomaly.api import app

    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "costsight"
    assert "stl" in body["detectors"]

    r = client.post("/generate", json={"n_days": 60, "seed": 99})
    assert r.status_code == 200
    body = r.json()
    assert body["n_days"] == 60
    assert body["n_services"] >= 5
    assert len(body["anomalies_injected"]) >= 3

    cur, _, _ = generate(n_days=60, seed=99)
    long = aggregate_by_service(cur)
    sample = long.head(200).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    payload = {"detector": "stl", "rows": sample.to_dict(orient="records")}
    r = client.post("/detect", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "stl"
    assert body["n_points"] == 200


def test_forecast_runs():
    cur, _, _ = generate(n_days=60, seed=11)
    long = aggregate_by_service(cur)
    fcast = forecast_per_service(long, horizon=7)
    assert {"date", "service", "kind", "cost", "lower", "upper"} <= set(fcast.columns)
    assert (fcast["kind"].isin(["history", "forecast"])).all()
    forecast_rows = fcast[fcast["kind"] == "forecast"]
    if not forecast_rows.empty:
        assert (forecast_rows["lower"] <= forecast_rows["upper"]).all()
        assert (forecast_rows["cost"] >= 0).all()

    proj = projected_monthly_spend(fcast)
    if not proj.empty:
        assert {"service", "projected_monthly", "daily_avg"} <= set(proj.columns)
        assert (proj["projected_monthly"] >= 0).all()
