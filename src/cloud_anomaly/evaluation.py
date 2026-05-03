"""Precision / Recall evaluation against the synthetic ground-truth labels.

Provides both an overall report and a per-anomaly-type breakdown — the latter
is the central research output of the project (Phase 2).

All entry points accept ``group_keys`` so the same code scores both the
legacy service-level setup and multi-granularity (service, env) runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def _resolve_keys(
    detections: pd.DataFrame, labels: pd.DataFrame, group_keys: Sequence[str] | None
) -> list[str]:
    """Pick the join keys: caller-supplied if given, else the intersection
    of (service, env) columns present in both frames, else just service.
    """
    if group_keys is not None:
        return list(group_keys)
    candidates = ["service", "env"]
    return [k for k in candidates if k in detections.columns and k in labels.columns] or [
        "service"
    ]


def evaluate(
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> Metrics:
    """Evaluate a detector run at the granularity of ``group_keys``.

    If ``group_keys`` is None the join keys are inferred from columns
    common to both frames (``service`` always, ``env`` if both carry it).
    """
    keys = _resolve_keys(detections, labels, group_keys)
    pred_df = detections[["date", *keys, "is_anomaly"]].rename(
        columns={"is_anomaly": "_pred"}
    )
    truth_df = labels[["date", *keys, "is_anomaly"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = pred_df.merge(truth_df, on=["date", *keys], how="left")
    merged["_truth"] = merged["_truth"].fillna(False).astype(bool)
    merged["_pred"] = merged["_pred"].astype(bool)

    pred = merged["_pred"]
    truth = merged["_truth"]
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return _metrics(tp, fp, fn)


def evaluate_by_type(
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-anomaly-type Precision/Recall — the headline result table."""
    keys = _resolve_keys(detections, labels, group_keys)
    pred_df = detections[["date", *keys, "is_anomaly"]].rename(
        columns={"is_anomaly": "_pred"}
    )
    truth_df = labels[["date", *keys, "is_anomaly", "anomaly_type"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = pred_df.merge(truth_df, on=["date", *keys], how="left")
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

    overall = evaluate(detections, labels, group_keys=keys)
    rows.append({"anomaly_type": "OVERALL", **overall.as_dict()})
    return pd.DataFrame(rows)


def compare_detectors(
    detector_outputs: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side per-anomaly-type comparison across detectors."""
    frames = []
    for name, det in detector_outputs.items():
        sub = evaluate_by_type(det, labels, group_keys=group_keys)
        sub.insert(0, "detector", name)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def evaluate_alerts(
    alerts: pd.DataFrame,
    labels: pd.DataFrame,
    group_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Quality of the *alerted* rows, broken down by severity band.

    Helpful for the FinOps-facing claim that HIGH-severity alerts should be
    overwhelmingly true positives. Returns one row per severity band with
    columns ``severity``, ``n_alerts``, ``true_positive``, ``precision``.
    """
    if alerts.empty:
        return pd.DataFrame(columns=["severity", "n_alerts", "true_positive", "precision"])

    keys = _resolve_keys(alerts, labels, group_keys)
    truth = labels[["date", *keys, "is_anomaly"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    merged = alerts.merge(truth, on=["date", *keys], how="left")
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
