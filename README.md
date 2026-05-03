# costsight — Automated Cloud Cost Anomaly Detection

Project 13 · Cloud Computing · Spring 2025–2026
**Furkan Can Karafil · Halil Utku Demirtaş**

[![CI](https://github.com/Urthella/costsight/actions/workflows/ci.yml/badge.svg)](https://github.com/Urthella/costsight/actions/workflows/ci.yml)

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

## Empirical results

Output of `python scripts/run_pipeline.py` on the default 90-day synthetic
dataset (seed = 42). Full table in [`examples/comparison.csv`](examples/comparison.csv).

### F1 by anomaly type

| Detector | Point spike | Level shift | Gradual drift | Overall |
|---|---:|---:|---:|---:|
| **Z-Score**         | **1.000** | 0.000 | 0.000 | 0.107 |
| **STL**             | 0.500 | **0.684** | **0.767** | **0.796** |
| **Isolation Forest**| 0.333 | 0.229 | 0.170 | 0.290 |

### Headline takeaways

- **No single method wins all anomaly types** — the central thesis of the
  project is empirically supported.
- **STL** is the strongest overall detector and handles trend-based
  anomalies (drift, level shift) cleanly.
- **Z-Score** is a perfect point-spike detector but completely blind to
  drift and level shifts, exactly as expected from a stationary baseline.
- **Isolation Forest** catches every point spike (recall = 1.0 there) but
  struggles to flag persistent shifts because they look "in distribution"
  once they stabilise — a known limitation of unsupervised tree models on
  univariate cost data.

## Running tests

```bash
pytest -q
```

## Scope

Phase 1 (May 20 deadline): synthetic data, three detectors, alert module,
dashboard, P/R evaluation. Out of scope: real-time streaming, multi-cloud,
production deployment, auto-remediation, cost forecasting.
