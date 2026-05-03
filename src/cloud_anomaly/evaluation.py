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
