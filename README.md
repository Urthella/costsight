# costsight — Automated Cloud Cost Anomaly Detection

Project 13 · Cloud Computing · Spring 2025–2026
**Furkan Can Karafil · Halil Utku Demirtaş**

[![CI](https://github.com/Urthella/costsight/actions/workflows/ci.yml/badge.svg)](https://github.com/Urthella/costsight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

End-to-end pipeline that ingests AWS CUR-style billing data, runs three
anomaly detectors in parallel (STL Decomposition, Isolation Forest, Z-Score)
at either **service** or **(service, env)** granularity, generates
severity-scored alerts, and visualizes everything in a Streamlit dashboard.

> 📄 **Full technical write-up:** [`REPORT.md`](REPORT.md) · 🎬 **Demo walkthrough:** [`DEMO.md`](DEMO.md) · 🎤 **Slide deck:** [`slides/deck.md`](slides/deck.md)

## Quick start

```bash
# 1. Install
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate synthetic data + run the full pipeline (service-level by default)
python scripts/run_pipeline.py

# 3. Re-run at (service, env) multi-granularity
python scripts/run_pipeline.py --granularity service_env

# 4. Launch the dashboard (Granularity radio in the sidebar)
streamlit run dashboard/app.py
```

Outputs land in `outputs/`:

- `detections_{detector}.csv` — per-day detector flags + scores
- `alerts_{detector}.{csv,json}` — severity-banded alert log
- `comparison.csv` — Precision / Recall / F1 by anomaly type, per detector
- `alert_quality.csv` — alert quality (true-positive rate) by severity band

To get statistically defensible numbers (mean ± std across 25 random seeds):

```bash
python scripts/run_benchmark.py --seeds 25 --granularity service
python scripts/run_benchmark.py --seeds 25 --granularity service_env
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
  detectors/           zscore, stl, iforest — common detect(df) interface
  alerts.py            severity = deviation × duration × $impact
  evaluation.py        Precision / Recall, per-anomaly-type, alert quality
  benchmark.py         multi-seed Monte Carlo runner
  pipeline.py          run() — wires everything together
dashboard/app.py       Streamlit UI (4 tabs)
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

| Type           | Default target          | Description                            | Example cause          |
|----------------|-------------------------|----------------------------------------|------------------------|
| Point spike    | EC2 prod, Lambda dev    | Single-day cost explosion              | Infinite loop          |
| Level shift   | RDS staging              | Persistent step up after change        | Mis-sized instances    |
| Gradual drift | S3 prod                  | Slow upward creep over a window        | Data accumulation      |

Each injected anomaly is recorded twice: in `data/raw/ground_truth_labels.csv`
at the (date, service) granularity (legacy schema, env-collapsed) and in
`data/raw/ground_truth_labels_granular.csv` at the (date, service, env)
granularity, so detectors can be scored fairly at either resolution.

## Detector outputs (common schema)

Every detector returns a frame with:

| column        | type     | meaning                           |
|---------------|----------|-----------------------------------|
| `date`        | datetime | day                               |
| `service`     | str      | AWS service name                  |
| `env`         | str      | environment (only when `group_keys` includes it) |
| `cost`        | float    | observed cost on that day         |
| `score`       | float    | anomaly score (higher = stranger) |
| `is_anomaly`  | bool     | flagged by the detector           |

All detectors share `detect(long_df, group_keys=("service",), ...)` so the
alert module, evaluation, and dashboard are detector-agnostic *and*
granularity-agnostic.

## Empirical results

Mean F1 ± std across **25 random seeds** (`python scripts/run_benchmark.py
--seeds 25 --granularity {service|service_env}`). Full tables in
[`examples/benchmark_summary_service.csv`](examples/benchmark_summary_service.csv)
and [`examples/benchmark_summary_service_env.csv`](examples/benchmark_summary_service_env.csv).

The synthetic CUR is split across prod / staging / dev environments and
each anomaly is injected into a *specific* env. Running at one
granularity vs. the other surfaces a clean tradeoff:

### F1 at the **service** granularity (legacy)

| Detector | Point spike | Level shift | Gradual drift | Overall |
|---|---:|---:|---:|---:|
| **Z-Score**         | **0.596 ± 0.143** | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.048 ± 0.017 |
| **STL**             | 0.300 ± 0.111 | 0.057 ± 0.052 | **0.615 ± 0.077** | **0.505 ± 0.058** |
| **Isolation Forest**| 0.160 ± 0.054 | **0.104 ± 0.045** | 0.159 ± 0.035 | 0.225 ± 0.032 |

### F1 at **(service, env)** multi-granularity

| Detector | Point spike | Level shift | Gradual drift | Overall |
|---|---:|---:|---:|---:|
| **Z-Score**         | **0.883 ± 0.156** | 0.008 ± 0.028 | 0.000 ± 0.000 | 0.092 ± 0.023 |
| **STL**             | 0.131 ± 0.026 | **0.152 ± 0.143** | **0.476 ± 0.051** | **0.484 ± 0.074** |
| **Isolation Forest**| 0.116 ± 0.010 | 0.058 ± 0.022 | 0.058 ± 0.019 | 0.136 ± 0.025 |

### Headline takeaways

- **No single method *or* granularity wins everything.** Z-Score's
  point-spike F1 jumps **+48%** (0.596 → 0.883) when env is broken
  out — a Lambda dev runaway loop is too small in absolute dollars to
  trip a service-level 3σ threshold but is obvious in dev's own
  series. Conversely STL's drift F1 *drops* −23% (0.615 → 0.476) at
  finer granularity because the drift lives in S3 prod (65% of S3
  spend), where pooling actually helps the trend detector.
- **STL** remains the strongest *overall* detector at both
  granularities — but its lead narrows when you cherry-pick the right
  granularity per anomaly type.
- **Z-Score** is a near-perfect point-spike detector (precision 0.99+)
  but completely blind to drift and level shifts, exactly as expected
  from a stationary baseline.
- **Isolation Forest** is mid-pack at service granularity and gets
  *worse* in multi-gran mode despite group-count-scaled contamination
  — a known regression that adaptive per-group thresholding (Phase 2+
  future work) should address.

## Running tests

```bash
pytest -q
```

## Scope

Phase 1 (May 20 deadline): synthetic data, three detectors, alert module,
dashboard, P/R evaluation. Phase 2 (this revision): multi-seed benchmark,
**(service, env) multi-granularity** pipeline + dashboard, env-aware
ground truth, granularity-tradeoff analysis. Out of scope: real-time
streaming, multi-cloud, production deployment, auto-remediation, cost
forecasting.

## License

[MIT](LICENSE) — see also [CONTRIBUTING.md](CONTRIBUTING.md) for how to
extend the project with new detectors or anomaly types.

## Authors

- **Furkan Can Karafil** ([@Urthella](https://github.com/Urthella)) · 222010020013
- **Halil Utku Demirtaş** · 222010020054
