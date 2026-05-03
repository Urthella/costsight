# Automated Cloud Cost Anomaly Detection

Project 13 · Cloud Computing · Spring 2025–2026
**Furkan Can Karafil · Halil Utku Demirtaş**

End-to-end pipeline that ingests AWS CUR-style billing data, runs three
anomaly detectors in parallel (STL Decomposition, Isolation Forest, Z-Score),
generates severity-scored alerts, and visualizes everything in a Streamlit
dashboard.

## Quick start

```bash
# 1. Install
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate synthetic data + run the full pipeline
python scripts/run_pipeline.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

Outputs land in `outputs/`:

- `detections_{detector}.csv` — per-day detector flags + scores
- `alerts_{detector}.{csv,json}` — severity-banded alert log
- `comparison.csv` — Precision / Recall / F1 by anomaly type, per detector

## Repository layout

```
src/cloud_anomaly/
  config.py            project constants (services, paths, severity bands)
  synthetic_data.py    AWS CUR-style data generator + ground-truth labels
  preprocessing.py     load, aggregate, pivot, gap-fill
  detectors/           zscore, stl, iforest — common detect(df) interface
  alerts.py            severity = deviation × duration × $impact
  evaluation.py        Precision / Recall, per-anomaly-type breakdown
  pipeline.py          run() — wires everything together
dashboard/app.py       Streamlit UI
scripts/run_pipeline.py  CLI entry point
tests/                 smoke tests
data/raw/              generated CUR + labels (gitignored)
outputs/               detector + alert + comparison artifacts (gitignored)
```

## Anomaly types injected

| Type           | Description                            | Example cause          |
|----------------|----------------------------------------|------------------------|
| Point spike    | Single-day cost explosion              | Infinite loop          |
| Level shift   | Persistent step up after change        | Mis-sized instances    |
| Gradual drift | Slow upward creep over a window        | Data accumulation      |

Each injected anomaly is recorded in `data/raw/ground_truth_labels.csv` so
detector outputs can be evaluated with real Precision / Recall numbers.

## Detector outputs (common schema)

Every detector returns a frame with:

| column        | type     | meaning                           |
|---------------|----------|-----------------------------------|
| `date`        | datetime | day                               |
| `service`     | str      | AWS service name                  |
| `cost`        | float    | observed cost on that day         |
| `score`       | float    | anomaly score (higher = stranger) |
| `is_anomaly`  | bool     | flagged by the detector           |

This is what makes the alert module and evaluation framework
detector-agnostic.

## Running tests

```bash
pytest -q
```

## Scope

Phase 1 (May 20 deadline): synthetic data, three detectors, alert module,
dashboard, P/R evaluation. Out of scope: real-time streaming, multi-cloud,
production deployment, auto-remediation, cost forecasting.
