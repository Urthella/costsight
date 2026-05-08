# costsight — Automated Cloud Cost Anomaly Detection

Project 13 · Cloud Computing · Spring 2025–2026
**Furkan Can Karafil · Halil Utku Demirtaş**

[![CI](https://github.com/Urthella/costsight/actions/workflows/ci.yml/badge.svg)](https://github.com/Urthella/costsight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

End-to-end pipeline that ingests AWS CUR-style billing data, runs three
anomaly detectors in parallel (STL Decomposition, Isolation Forest, Z-Score),
generates severity-scored alerts, and visualizes everything in a Streamlit
dashboard.

> 📄 **Full technical write-up:** [`REPORT.md`](REPORT.md) · 🎬 **Demo walkthrough:** [`DEMO.md`](DEMO.md) · 🎤 **Slide deck:** [`slides/deck.md`](slides/deck.md)

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
- `attribution_{detector}.csv` — root-cause hint per alert (which region / usage_type drove the spend)
- `comparison.csv` — Precision / Recall / F1 by anomaly type, per detector
- `alert_quality.csv` — alert quality (true-positive rate) by severity band

To get statistically defensible numbers (mean ± std across 25 random seeds):

```bash
python scripts/run_benchmark.py --seeds 25
```

To re-render the presentation figures from a fresh run:

```bash
python scripts/make_figures.py    # writes slides/figures/*.png
```

## Repository layout

```
src/cloud_anomaly/
  config.py            project constants (services, paths, severity bands)
  synthetic_data.py    AWS CUR-style data generator + ground-truth labels
  preprocessing.py     load, aggregate, pivot, gap-fill
  detectors/           zscore, stl, iforest, ensemble — common detect(df) interface
  alerts.py            severity = deviation × duration × $impact
  attribution.py       root-cause hint per alert (region / usage_type)
  evaluation.py        Precision / Recall, alert quality, TTD,
                       cost-saved estimate, bootstrap CI, Wilcoxon test
  forecast.py          Holt-Winters per-service forecast + projection
  theoretical_scores.py proposal a-priori ratings (radar charts)
  benchmark.py         multi-seed Monte Carlo runner
  pipeline.py          run() — wires everything together
dashboard/app.py       Streamlit UI (9 tabs: cost trend / alert log /
                       root-cause / detector comparison / calendar /
                       forecast / lab / replay / raw data)
scripts/
  run_pipeline.py        single-run CLI
  run_benchmark.py       25-seed CLI
  make_figures.py        renders presentation PNGs
slides/
  deck.md                Marp slide deck (renders to PDF/HTML)
  SLIDE_UPDATES.md       per-slide guide for the existing deck
  figures/               4 ready-to-use 16:9 PNGs
examples/                committed sample artifacts
tests/                   smoke tests, run on every CI commit
.github/workflows/ci.yml CI: pytest + pipeline on Python 3.11 and 3.12
data/raw/                generated CUR + labels (gitignored)
outputs/                 run artifacts (gitignored)
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

Mean ± std across **25 random seeds** (`python scripts/run_benchmark.py
--seeds 25`). Full table in [`examples/benchmark_summary.csv`](examples/benchmark_summary.csv).

### F1 by anomaly type

| Detector | Point spike | Level shift | Gradual drift | Overall |
|---|---:|---:|---:|---:|
| **Z-Score**         | **0.962 ± 0.078** | 0.012 ± 0.033 | 0.000 ± 0.000 | 0.105 ± 0.018 |
| **STL**             | 0.522 ± 0.082 | **0.616 ± 0.204** | **0.734 ± 0.052** | **0.757 ± 0.064** |
| **Isolation Forest**| 0.247 ± 0.035 | 0.216 ± 0.060 | 0.217 ± 0.034 | 0.319 ± 0.036 |

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

## Root-cause attribution

For every alert the pipeline produces a one-line, human-readable hint
about *which* CUR dimension drove the spend above its 14-day baseline:

> *EC2 spend on 2025-03-19 is $957 (+391% vs 14-day baseline);
> us-east-1 region drove 100% of the increase.*

Attribution is computed per (date, service) by decomposing the spend
along `region` and `usage_type`, comparing against the trailing
14-day per-value baseline, and reporting the dimension+value that
contributed most to the anomaly delta. Available in
[`outputs/attribution_{detector}.csv`](examples/attribution_stl_sample.csv)
and on the dashboard's *Root-cause* tab.

This is a Level-1-friendly take on the Level-2 "root-cause attribution"
deliverable — concise, deterministic, and immediately useful for FinOps
triage.

## Running tests

```bash
pytest -q
```

## Deploying the dashboard

The Streamlit dashboard is one-click deployable to **Streamlit Community
Cloud** — the easiest path to a live URL for the demo.

1. Sign in at <https://streamlit.io/cloud> with your GitHub account.
2. Click **New app**, point it at this repository, branch `main`,
   main file path: `dashboard/app.py`.
3. Python version: **3.11**. The platform installs everything from
   `requirements.txt` automatically; no extra config is needed.
4. Once it builds (~3 min), Streamlit publishes a public URL of the form
   `https://<app-name>.streamlit.app`. Share it during the demo.

`.streamlit/config.toml` is committed and pre-configures the dark theme
and the brand color, so the deployed instance looks identical to local.

For a containerized deploy (ECS, Cloud Run, Fly.io, Render), see
[`REPORT.md` § Cloud architecture](REPORT.md#cloud-architecture-production-path).

## Scope

Phase 1 (May 20 deadline): synthetic data, three detectors plus an
ensemble vote, alert module, root-cause attribution, P/R evaluation,
multi-seed benchmark, dashboard with calendar / forecast / lab /
replay tabs, statistical significance tests. Phase 2 (post-finals):
comparison report extension, paper-style writeup. Out of scope:
real-time streaming, multi-cloud ingestion, production deployment of
the detection pipeline (the dashboard is deployable; the pipeline
remains batch).

## License

[MIT](LICENSE) — see also [CONTRIBUTING.md](CONTRIBUTING.md) for how to
extend the project with new detectors or anomaly types.

## Authors

- **Furkan Can Karafil** ([@Urthella](https://github.com/Urthella)) · 222010020013
- **Halil Utku Demirtaş** · 222010020054
