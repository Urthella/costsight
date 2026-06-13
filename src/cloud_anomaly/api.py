"""FastAPI REST surface for the cloud anomaly pipeline.

Run locally:
    uvicorn cloud_anomaly.api:app --reload --port 8000

Docker (preferred for cloud deploys):
    docker compose up api

Endpoints:
    GET  /health           - liveness probe
    GET  /                 - service metadata
    POST /generate         - produce a fresh synthetic dataset
    POST /detect           - run a single detector on supplied long-format JSON
    POST /alerts           - convenience wrapper: detect + build_alerts + attribute
    GET  /metrics          - multi-detector P/R/F1 against the ground truth in raw_dir
    GET  /forecast         - Holt-Winters per-service forecast for the next N days
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .alerts import build_alerts
from .attribution import attribute
from .carbon import carbon_footprint
from .clustering import cluster_alerts, summarize_incidents
from .config import RAW_DIR
from .detectors import DETECTORS
from .detectors.ensemble import detect as ensemble_detect
from .drift import detect_drift
from .evaluation import (
    compare_detectors,
    cost_saved_estimate,
    evaluate_by_type,
)
from .forecast import forecast_per_service, projected_monthly_spend
from .playbook import PLAYBOOKS
from .preprocessing import aggregate_by_service, aggregate_daily, load_cur
from .recommender import all_recommendations
from .synthetic_data import SCENARIOS, generate
from .tag_governance import evaluate_tagging

app = FastAPI(
    title="costsight API",
    description="REST surface for cloud cost anomaly detection (Project 13).",
    version="1.1.0",
)

# The React frontend is served from a different origin in dev (Vite :5173)
# and may be a separate static host in prod, so allow cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CostRow(BaseModel):
    date: str
    service: str
    cost: float


class DetectRequest(BaseModel):
    detector: Literal["zscore", "stl", "iforest", "ensemble"] = "stl"
    rows: list[CostRow] = Field(..., description="Long-format daily rows (date, service, cost).")


class GenerateRequest(BaseModel):
    n_days: int = Field(90, ge=30, le=365)
    seed: int = Field(42, ge=0)


class ForecastRequest(BaseModel):
    horizon_days: int = Field(14, ge=1, le=60)
    seed: int = Field(0, ge=0)


def _rows_to_long(rows: list[CostRow]) -> pd.DataFrame:
    if not rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_long_or_404() -> pd.DataFrame:
    parquet = RAW_DIR / "cur_synthetic.parquet"
    if not parquet.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset on disk. Call POST /generate first or run "
                "scripts/run_pipeline.py."
            ),
        )
    cur_df = load_cur()
    return aggregate_by_service(cur_df), cur_df


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "costsight",
        "version": app.version,
        "detectors": list(DETECTORS.keys()),
        "endpoints": [
            "GET /health", "POST /generate", "POST /detect", "POST /alerts",
            "GET /metrics", "GET /forecast",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def http_generate(req: GenerateRequest) -> dict[str, Any]:
    cur, labels, anomalies = generate(n_days=req.n_days, seed=req.seed)
    return {
        "n_days": req.n_days,
        "seed": req.seed,
        "n_rows": int(len(cur)),
        "n_services": int(cur["service"].nunique()),
        "anomalies_injected": [
            {
                "service": a.service, "type": a.anomaly_type,
                "start_day": a.start_day, "end_day": a.end_day,
                "multiplier": a.multiplier,
            }
            for a in anomalies
        ],
        "ground_truth_rows": int(labels["is_anomaly"].sum()),
    }


@app.post("/detect")
def http_detect(req: DetectRequest) -> dict[str, Any]:
    long = _rows_to_long(req.rows)
    fn = DETECTORS[req.detector]
    detections = fn(long)
    return {
        "detector": req.detector,
        "n_points": int(len(detections)),
        "n_flagged": int(detections["is_anomaly"].sum()),
        "detections": detections.assign(
            date=detections["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }


@app.post("/alerts")
def http_alerts(req: DetectRequest) -> dict[str, Any]:
    long = _rows_to_long(req.rows)
    detections = DETECTORS[req.detector](long)
    alerts = build_alerts(detections, detector_name=req.detector,
                          dataset_days=int(long["date"].nunique()))
    if alerts.empty:
        return {"detector": req.detector, "n_alerts": 0, "alerts": []}
    cur_df = long.assign(region="-", usage_type="-")[
        ["date", "service", "region", "usage_type", "cost"]
    ]
    attribution_df = attribute(cur_df, alerts)
    payload = alerts.copy()
    payload["date"] = payload["date"].dt.strftime("%Y-%m-%d")
    return {
        "detector": req.detector,
        "n_alerts": int(len(alerts)),
        "alerts": payload.to_dict(orient="records"),
        "attribution": (
            attribution_df.assign(date=attribution_df["date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
            if not attribution_df.empty else []
        ),
    }


@app.get("/metrics")
def http_metrics() -> dict[str, Any]:
    long, _ = _load_long_or_404()
    labels_path = RAW_DIR / "ground_truth_labels.csv"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="ground-truth labels not found")
    labels = pd.read_csv(labels_path, parse_dates=["date"])
    detector_outputs = {name: fn(long) for name, fn in DETECTORS.items()}
    comparison = compare_detectors(detector_outputs, labels)
    return {"comparison": comparison.round(4).to_dict(orient="records")}


@app.get("/forecast")
def http_forecast(horizon_days: int = 14, seed: int = 0) -> dict[str, Any]:
    long, _ = _load_long_or_404()
    fcast = forecast_per_service(long, horizon=horizon_days, rng_seed=seed)
    fcast_records = fcast.copy()
    fcast_records["date"] = pd.to_datetime(fcast_records["date"]).dt.strftime("%Y-%m-%d")
    proj = projected_monthly_spend(fcast)
    return {
        "horizon_days": horizon_days,
        "forecast": fcast_records.to_dict(orient="records"),
        "projected_monthly": proj.to_dict(orient="records") if not proj.empty else [],
    }


# --------------------------------------------------------------------------- #
# Snapshot API consumed by the React frontend                                  #
# --------------------------------------------------------------------------- #

def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame to JSON records, ISO-formatting any datetime cols."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@lru_cache(maxsize=16)
def build_snapshot(scenario: str = "default", n_days: int = 90, seed: int = 42) -> dict[str, Any]:
    """Run the full pipeline once and return everything the dashboard renders.

    A single round-trip the React app caches per (scenario, n_days, seed) and
    fans out across ~15 views. Heavy/optional work (perf grid, LLM explain)
    stays in its own lazy endpoint. Memoized server-side so repeat visits to
    the same (scenario, n_days, seed) are instant.
    """
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"unknown scenario '{scenario}'")

    cur_df, labels_df, _ = generate(n_days=n_days, seed=seed, scenario=scenario)
    long = aggregate_by_service(cur_df)
    daily = aggregate_daily(cur_df)
    dataset_days = int(long["date"].nunique())

    # Run the three base detectors once, then derive the ensemble from them —
    # otherwise the ensemble re-runs Isolation Forest (the slow one) a 2nd time.
    base = {name: DETECTORS[name](long) for name in ("zscore", "stl", "iforest") if name in DETECTORS}
    detections = {}
    for name in DETECTORS:
        detections[name] = (
            ensemble_detect(long, base=base) if name == "ensemble" else base[name]
        )
    alerts_by = {
        name: build_alerts(det, name, dataset_days=dataset_days)
        for name, det in detections.items()
    }
    all_alerts = pd.concat(
        [a for a in alerts_by.values() if not a.empty], ignore_index=True
    ) if any(not a.empty for a in alerts_by.values()) else pd.DataFrame()

    # Root-cause attribution on the union of alerts.
    try:
        attribution = attribute(cur_df, all_alerts) if not all_alerts.empty else pd.DataFrame()
    except Exception:
        attribution = pd.DataFrame()

    comparison = compare_detectors(detections, labels_df)

    # Best detector by overall F1 -> cost-saved estimate (mirrors the dashboard).
    best_name, best_f1 = None, -1.0
    for name, det in detections.items():
        f1 = float(evaluate_by_type(det, labels_df).iloc[-1]["f1"])
        if f1 > best_f1:
            best_f1, best_name = f1, name
    saved = (
        cost_saved_estimate(cur_df, detections[best_name], labels_df)
        if best_name else {"saved": 0.0, "total_anomaly_cost": 0.0, "ratio": 0.0}
    )

    def _safe(fn, fallback):
        try:
            return fn()
        except Exception:
            return fallback

    carbon = _safe(lambda: carbon_footprint(cur_df), None)
    carbon_block = {
        "kg_co2": round(float(carbon.kg_co2), 2),
        "km_driven_equiv": round(float(carbon.km_driven_equiv), 1),
        "tree_years_equiv": round(float(carbon.tree_years_equiv), 2),
        "cost_usd": round(float(carbon.cost_usd), 2),
        "by_service": _df_records(carbon.by_service),
        "by_region": _df_records(carbon.by_region),
    } if carbon is not None else {}

    recs = _safe(lambda: all_recommendations(cur_df), pd.DataFrame())
    tagging = _safe(lambda: evaluate_tagging(cur_df), None)
    tagging_block = {
        "debt_usd": round(float(tagging.debt_usd), 2),
        "coverage": _df_records(tagging.coverage),
        "worst_services": _df_records(tagging.worst_services),
        "policy_yaml": tagging.policy_yaml,
    } if tagging is not None else {}

    drift = _safe(lambda: detect_drift(long), pd.DataFrame())

    incidents = pd.DataFrame()
    if not all_alerts.empty:
        clustered = _safe(lambda: cluster_alerts(all_alerts), pd.DataFrame())
        if not clustered.empty:
            incidents = _safe(lambda: summarize_incidents(clustered), pd.DataFrame())

    fcast = _safe(lambda: forecast_per_service(long, horizon=14, rng_seed=0), pd.DataFrame())
    proj = _safe(lambda: projected_monthly_spend(fcast), pd.DataFrame()) if not fcast.empty else pd.DataFrame()

    per_detector_flags = {n: int(d["is_anomaly"].sum()) for n, d in detections.items()}
    consensus = 0
    if not all_alerts.empty:
        per_pair = all_alerts.groupby(["date", "service"])["detector"].nunique()
        consensus = int((per_pair >= 2).sum())

    return {
        "meta": {
            "scenario": scenario,
            "n_days": n_days,
            "seed": seed,
            "dataset_days": dataset_days,
            "n_services": int(cur_df["service"].nunique()),
            "services": sorted(cur_df["service"].unique().tolist()),
            "total_spend": round(float(cur_df["cost"].sum()), 2),
        },
        "kpis": {
            "total_spend": round(float(cur_df["cost"].sum()), 2),
            "n_services": int(cur_df["service"].nunique()),
            "flags_per_detector": per_detector_flags,
            "total_flags": int(sum(per_detector_flags.values())),
            "consensus_alerts": consensus,
            "best_detector": best_name,
            "savable_usd": round(float(saved["saved"]), 2),
            "leak_ratio": round(float(saved["ratio"]), 4),
            "total_leak_usd": round(float(saved["total_anomaly_cost"]), 2),
        },
        "detectors": list(DETECTORS.keys()),
        "daily": _df_records(daily),
        "series": _df_records(long),
        "detections": {name: _df_records(det) for name, det in detections.items()},
        "alerts": _df_records(all_alerts),
        "attribution": _df_records(attribution),
        "comparison": _df_records(comparison),
        "carbon": carbon_block,
        "recommendations": _df_records(recs),
        "tagging": tagging_block,
        "drift": _df_records(drift),
        "incidents": _df_records(incidents),
        "forecast": _df_records(fcast),
        "projected_monthly": _df_records(proj),
        "ground_truth": _df_records(labels_df),
        "playbooks": PLAYBOOKS,
    }


@app.get("/api/scenarios")
def http_scenarios() -> dict[str, Any]:
    return {"scenarios": [{"key": k, "description": v} for k, v in SCENARIOS.items()]}


@app.get("/api/snapshot")
def http_snapshot(scenario: str = "default", n_days: int = 90, seed: int = 42) -> dict[str, Any]:
    # Positional call so it shares one lru_cache key with the startup warm-up.
    return build_snapshot(scenario, n_days, seed)


@app.on_event("startup")
def _warm_default_snapshot() -> None:
    """Precompute the default snapshot in the background so the very first
    page load is instant instead of waiting on the cold pipeline (~4s)."""
    def _go() -> None:
        try:
            build_snapshot("default", 90, 42)
        except Exception:  # noqa: BLE001 — warm-up is best-effort
            pass

    threading.Thread(target=_go, daemon=True).start()


@app.get("/api/perf")
def http_perf(scenario: str = "default", n_days: int = 90, seed: int = 42) -> dict[str, Any]:
    """Detector runtime benchmark grid (slow; kept off the snapshot path)."""
    from .perf import benchmark_grid

    grid = benchmark_grid(seed=seed)
    return {"perf": _df_records(grid)}


class ExplainRequest(BaseModel):
    service: str
    date: str
    severity: str = "HIGH"
    cost: float = 0.0
    flagged_by: str = ""
    top_dimension: str = ""
    top_value: str = ""


@app.post("/api/explain")
def http_explain(req: ExplainRequest) -> dict[str, Any]:
    """On-demand LLM (or templated) root-cause explanation for one alert."""
    from .explainer import explain_alert

    alert_row = {
        "date": pd.Timestamp(req.date), "service": req.service,
        "severity": req.severity, "cost": req.cost, "flagged_by": req.flagged_by,
    }
    attr_row = {"top_dimension": req.top_dimension, "top_value": req.top_value}
    text = explain_alert(alert_row, attr_row)
    return {"explanation": text}
