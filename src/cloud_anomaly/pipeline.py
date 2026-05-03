"""End-to-end pipeline: generate (or load) data → detect → alert → evaluate."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alerts import build_alerts, write_alerts
from .attribution import attribute
from .config import OUTPUTS_DIR, RAW_DIR
from .detectors import DETECTORS
from .evaluation import compare_detectors, evaluate_alerts
from .preprocessing import aggregate_by_service, load_cur
from .synthetic_data import generate


def run(
    regenerate: bool = True,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full pipeline; returns a dict of intermediate artifacts."""
    out_dir = out_dir or OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if regenerate or not (RAW_DIR / "cur_synthetic.parquet").exists():
        cur_df, labels_df, _ = generate()
    else:
        cur_df = load_cur()
        labels_df = pd.read_csv(RAW_DIR / "ground_truth_labels.csv", parse_dates=["date"])

    long = aggregate_by_service(cur_df)

    detector_outputs: dict[str, pd.DataFrame] = {}
    alerts_by_detector: dict[str, pd.DataFrame] = {}
    attributions_by_detector: dict[str, pd.DataFrame] = {}

    for name, fn in DETECTORS.items():
        detections = fn(long)
        detector_outputs[name] = detections
        detections_csv = out_dir / f"detections_{name}.csv"
        detections.to_csv(detections_csv, index=False)

        alerts = build_alerts(detections, detector_name=name, dataset_days=long["date"].nunique())
        alerts_by_detector[name] = alerts
        write_alerts(alerts, name, out_dir=out_dir)

        attribution_df = attribute(cur_df, alerts)
        attributions_by_detector[name] = attribution_df
        if not attribution_df.empty:
            attribution_df.to_csv(out_dir / f"attribution_{name}.csv", index=False)

    comparison = compare_detectors(detector_outputs, labels_df)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    alert_quality_rows = []
    for name, alerts_df in alerts_by_detector.items():
        sub = evaluate_alerts(alerts_df, labels_df)
        if sub.empty:
            continue
        sub.insert(0, "detector", name)
        alert_quality_rows.append(sub)
    alert_quality = (
        pd.concat(alert_quality_rows, ignore_index=True)
        if alert_quality_rows
        else pd.DataFrame()
    )
    if not alert_quality.empty:
        alert_quality.to_csv(out_dir / "alert_quality.csv", index=False)

    return {
        "cur": cur_df,
        "labels": labels_df,
        "long": long,
        "detections": detector_outputs,
        "alerts": alerts_by_detector,
        "attributions": attributions_by_detector,
        "comparison": comparison,
        "alert_quality": alert_quality,
    }


if __name__ == "__main__":
    artifacts = run()
    print("\n=== Detector comparison (P/R by anomaly type) ===")
    print(artifacts["comparison"].to_string(index=False))
    if not artifacts["alert_quality"].empty:
        print("\n=== Alert quality by severity band ===")
        print(artifacts["alert_quality"].to_string(index=False))
    print("\nAlerts written under outputs/")
