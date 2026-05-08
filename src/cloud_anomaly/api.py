"""FastAPI REST surface for the cloud anomaly pipeline.

Run locally:
    uvicorn cloud_anomaly.api:app --reload --port 8000

Docker (preferred for cloud deploys):
    docker compose up api

Endpoints:
    GET  /health           — liveness probe
    GET  /                 — service metadata
    POST /generate         — produce a fresh synthetic dataset
    POST /detect           — run a single detector on supplied long-format JSON
    POST /alerts           — convenience wrapper: detect + build_alerts + attribute
    GET  /metrics          — multi-detector P/R/F1 against the ground truth in raw_dir
    GET  /forecast         — Holt-Winters per-service forecast for the next N days
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .alerts import build_alerts
from .attribution import attribute
from .config import RAW_DIR
from .detectors import DETECTORS
from .evaluation import compare_detectors
from .forecast import forecast_per_service, projected_monthly_spend
from .preprocessing import aggregate_by_service, load_cur
from .synthetic_data import generate

app = FastAPI(
    title="costsight API",
    description="REST surface for cloud cost anomaly detection (Project 13).",
    version="1.0.0",
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
    cur_df = long.assign(region="—", usage_type="—")[
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
