"""Precision / Recall evaluation against the synthetic ground-truth labels.

Provides both an overall report and a per-anomaly-type breakdown — the latter
is the central research output of the project (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def _metrics(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def evaluate(detections: pd.DataFrame, labels: pd.DataFrame) -> Metrics:
    """Evaluate a detector run against ground truth at (date, service) granularity."""
    pred_df = detections[["date", "service", "is_anomaly"]].rename(
        columns={"is_anomaly": "_pred"}
    )
    truth_df = labels[["date", "service", "is_anomaly"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = pred_df.merge(truth_df, on=["date", "service"], how="left")
    merged["_truth"] = merged["_truth"].fillna(False).astype(bool)
    merged["_pred"] = merged["_pred"].astype(bool)

    pred = merged["_pred"]
    truth = merged["_truth"]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return _metrics(tp, fp, fn)


def evaluate_by_type(detections: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per-anomaly-type Precision/Recall — the headline result table."""
    pred_df = detections[["date", "service", "is_anomaly"]].rename(
        columns={"is_anomaly": "_pred"}
    )
    truth_df = labels[["date", "service", "is_anomaly", "anomaly_type"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = pred_df.merge(truth_df, on=["date", "service"], how="left")
    merged["_pred"] = merged["_pred"].astype(bool)
    merged["_truth"] = merged["_truth"].fillna(False).astype(bool)
    merged["anomaly_type"] = merged["anomaly_type"].fillna("")

    rows = []
    for anomaly_type in sorted(t for t in merged["anomaly_type"].unique() if t):
        type_truth = (merged["anomaly_type"] == anomaly_type) & merged["_truth"]
        tp = int((merged["_pred"] & type_truth).sum())
        fn = int((~merged["_pred"] & type_truth).sum())
        # FP = predicted anomalies where the row is not anomalous (any type).
        fp = int((merged["_pred"] & ~merged["_truth"]).sum())
        m = _metrics(tp, fp, fn)
        rows.append({"anomaly_type": anomaly_type, **m.as_dict()})

    overall = evaluate(detections, labels)
    rows.append({"anomaly_type": "OVERALL", **overall.as_dict()})
    return pd.DataFrame(rows)


def compare_detectors(
    detector_outputs: dict[str, pd.DataFrame], labels: pd.DataFrame
) -> pd.DataFrame:
    """Side-by-side per-anomaly-type comparison across detectors."""
    frames = []
    for name, det in detector_outputs.items():
        sub = evaluate_by_type(det, labels)
        sub.insert(0, "detector", name)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def time_to_detect(
    detections: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Days from anomaly onset to first detection, per ground-truth anomaly window.

    Returns one row per (anomaly_window_start, service, anomaly_type) with the
    column ``days_to_detect`` (NaN if never detected within the window).
    """
    truth = labels[labels["is_anomaly"]].copy()
    if truth.empty:
        return pd.DataFrame(columns=[
            "service", "anomaly_type", "window_start", "window_end", "days_to_detect",
        ])
    truth["date"] = pd.to_datetime(truth["date"])
    truth = truth.sort_values(["service", "anomaly_type", "date"])
    # Group consecutive truth dates per (service, anomaly_type) into windows.
    truth["gap"] = (
        truth.groupby(["service", "anomaly_type"])["date"].diff().dt.days.fillna(1) > 1
    )
    truth["window_id"] = truth.groupby(["service", "anomaly_type"])["gap"].cumsum()
    windows = truth.groupby(["service", "anomaly_type", "window_id"]).agg(
        window_start=("date", "min"),
        window_end=("date", "max"),
    ).reset_index().drop(columns="window_id")

    flagged = detections[detections["is_anomaly"]].copy()
    flagged["date"] = pd.to_datetime(flagged["date"])

    rows = []
    for _, w in windows.iterrows():
        mask = (
            (flagged["service"] == w["service"])
            & (flagged["date"] >= w["window_start"])
            & (flagged["date"] <= w["window_end"])
        )
        first = flagged.loc[mask, "date"].min() if mask.any() else pd.NaT
        ttd = (first - w["window_start"]).days if pd.notna(first) else float("nan")
        rows.append({
            "service": w["service"],
            "anomaly_type": w["anomaly_type"],
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "days_to_detect": ttd,
        })
    return pd.DataFrame(rows)


def cost_saved_estimate(
    cur_df: pd.DataFrame,
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    response_lag_days: int = 1,
) -> dict[str, float]:
    """Rough $ savings if anomalies had been acted on `response_lag_days` after detection.

    For each (service, anomaly_window):
      - if the detector flagged within the window, savings = sum of cost on the
        days between (first_detection + lag) and window_end, MINUS the per-day
        baseline (14-day rolling mean before window_start).
      - if never flagged, savings = 0.
    Returns dict with total $ saved, $ wasted (truth $ minus saved), and ratio.
    """
    cur = cur_df[["date", "service", "cost"]].copy()
    cur["date"] = pd.to_datetime(cur["date"])
    daily = cur.groupby(["date", "service"], as_index=False)["cost"].sum()

    truth = labels[labels["is_anomaly"]].copy()
    if truth.empty:
        return {"saved": 0.0, "total_anomaly_cost": 0.0, "ratio": 0.0}
    truth["date"] = pd.to_datetime(truth["date"])
    truth = truth.sort_values(["service", "anomaly_type", "date"])
    truth["gap"] = (
        truth.groupby(["service", "anomaly_type"])["date"].diff().dt.days.fillna(1) > 1
    )
    truth["window_id"] = truth.groupby(["service", "anomaly_type"])["gap"].cumsum()
    windows = truth.groupby(["service", "anomaly_type", "window_id"]).agg(
        window_start=("date", "min"),
        window_end=("date", "max"),
    ).reset_index()

    flagged = detections[detections["is_anomaly"]].copy()
    flagged["date"] = pd.to_datetime(flagged["date"])

    saved_total = 0.0
    anomaly_cost_total = 0.0

    for _, w in windows.iterrows():
        baseline_window = daily[
            (daily["service"] == w["service"])
            & (daily["date"] >= w["window_start"] - pd.Timedelta(days=14))
            & (daily["date"] < w["window_start"])
        ]
        baseline_per_day = baseline_window["cost"].mean() if not baseline_window.empty else 0.0

        in_window = daily[
            (daily["service"] == w["service"])
            & (daily["date"] >= w["window_start"])
            & (daily["date"] <= w["window_end"])
        ]
        excess_per_day = (in_window["cost"] - baseline_per_day).clip(lower=0)
        anomaly_cost_total += float(excess_per_day.sum())

        mask = (
            (flagged["service"] == w["service"])
            & (flagged["date"] >= w["window_start"])
            & (flagged["date"] <= w["window_end"])
        )
        if not mask.any():
            continue
        first_flag = flagged.loc[mask, "date"].min()
        action_day = first_flag + pd.Timedelta(days=response_lag_days)
        savable = in_window[in_window["date"] >= action_day]
        savable_excess = (savable["cost"] - baseline_per_day).clip(lower=0)
        saved_total += float(savable_excess.sum())

    ratio = saved_total / anomaly_cost_total if anomaly_cost_total else 0.0
    return {
        "saved": saved_total,
        "total_anomaly_cost": anomaly_cost_total,
        "ratio": ratio,
    }


def evaluate_alerts(
    alerts: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Quality of the *alerted* (date, service) pairs, broken down by severity.

    Helpful for the FinOps-facing claim that HIGH-severity alerts should be
    overwhelmingly true positives. Returns one row per severity band with
    columns ``severity``, ``n_alerts``, ``true_positive``, ``precision``.
    """
    if alerts.empty:
        return pd.DataFrame(columns=["severity", "n_alerts", "true_positive", "precision"])

    truth = labels[["date", "service", "is_anomaly"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = alerts.merge(truth, on=["date", "service"], how="left")
    merged["_truth"] = merged["_truth"].fillna(False).astype(bool)

    rows = []
    for severity in ["HIGH", "MEDIUM", "LOW"]:
        sub = merged[merged["severity"] == severity]
        n = len(sub)
        if n == 0:
            continue
        tp = int(sub["_truth"].sum())
        rows.append(
            {
                "severity": severity,
                "n_alerts": n,
                "true_positive": tp,
                "precision": round(tp / n, 4) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)
