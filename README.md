# costsight — Automated Cloud Cost Anomaly Detection

Project 13 · Cloud Computing · Spring 2025–2026
**Furkan Can Karafil · Halil Utku Demirtaş**

[![CI](https://github.com/Urthella/costsight/actions/workflows/ci.yml/badge.svg)](https://github.com/Urthella/costsight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

End-to-end pipeline that ingests AWS CUR-style billing data, runs four
anomaly detectors in parallel (STL Decomposition, Isolation Forest, Z-Score,
Ensemble), generates severity-scored alerts, and serves everything through a
**FastAPI** backend to a **React** web app (Vite + TypeScript + Tailwind +
Plotly).

> 📄 **Full technical write-up:** [`REPORT.md`](REPORT.md) · 🎬 **Demo walkthrough:** [`DEMO.md`](DEMO.md) · 🎤 **Slide deck:** [`slides/deck.md`](slides/deck.md)

## Quick start

```bash
# 1. Install backend
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate synthetic data + run the full pipeline (optional; the API can also generate on demand)
python scripts/run_pipeline.py

# 3. Start the API (data source for the web app)
uvicorn cloud_anomaly.api:app --reload --port 8000

# 4. In a second terminal, start the web app
cd frontend
npm install                        # first run only
npm run dev                        # http://localhost:5173 (proxies /api -> :8000)
```

> 💡 **Run it on your own bill:** point the pipeline / API at a real AWS Cost &
> Usage Report via `cloud_anomaly.cur_loader.load_cur_csv()`; detection,
> alerts, attribution, forecast, carbon and recommendations all run against
> your data. Only the ground-truth-dependent Precision/Recall view stays blank,
> since real billing data ships no anomaly labels.
>
> The previous Streamlit UI is archived under [`legacy/`](legacy/) (git tag
> `streamlit-v1`).

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
  api.py               FastAPI: /api/snapshot (full bundle) + scenarios/perf/explain
frontend/              React + Vite + TS + Tailwind + Plotly web app (19 views:
                       summary / cost trend / calendar / alert log / root-cause /
                       detector comparison / incidents / drift / forecast / budget /
                       recommendations / playbook / carbon / tagging / AI explain /
                       perf / lab / replay / raw data). Reads /api/snapshot.
legacy/                archived Streamlit app (pre-React; git tag streamlit-v1)
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

## Deploying the app

The frontend is a static bundle and the backend is a stateless API, so they
deploy independently:

1. **Backend** — containerize with the included `Dockerfile` (serves
   `uvicorn cloud_anomaly.api:app` on :8000) and run it anywhere (ECS, Cloud
   Run, Fly.io, Render). See
   [`REPORT.md` § Cloud architecture](REPORT.md#cloud-architecture-production-path).
2. **Frontend** — `cd frontend && npm run build` produces `frontend/dist/`,
   a static site deployable to Vercel / Netlify / S3+CloudFront / GitHub Pages.
   Set `VITE_API_URL` to the deployed API origin at build time; in dev it
   proxies `/api` to `:8000` automatically.

### Docker (full stack)

```bash
docker compose up --build          # web app on :8080, REST API on :8000
```

Two services off the compose file:

- `api` — FastAPI (`http://localhost:8000`, OpenAPI at `/docs`); mounts
  `./data` and `./outputs` so artifacts survive restarts.
- `frontend` — the React app built and served by nginx (`http://localhost:8080`),
  which proxies `/api` to the `api` service.

### REST API (FastAPI)

The same detection pipeline is also exposed as an HTTP service so it can
sit behind API Gateway / ALB in a real cloud deploy.

```bash
uvicorn cloud_anomaly.api:app --reload --port 8000
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`     | Liveness probe |
| GET  | `/`           | Service metadata + detector list |
| POST | `/generate`   | Produce a synthetic dataset (n_days, seed) |
| POST | `/detect`     | Run a detector on supplied long-format JSON |
| POST | `/alerts`     | Detect → severity-band → root-cause attribution |
| GET  | `/metrics`    | Multi-detector P/R/F1 against on-disk ground truth |
| GET  | `/forecast`   | Holt-Winters per-service forecast (horizon_days) |

Browse the auto-generated OpenAPI docs at `/docs` (Swagger UI) or `/redoc`.

### Continuous benchmarking

`.github/workflows/benchmark.yml` re-runs the 25-seed Monte Carlo every
Sunday at 02:00 UTC and uploads `outputs/benchmark_summary.csv`,
`outputs/benchmark_raw.csv`, and the regenerated presentation figures
as a 90-day-retained workflow artifact. Trigger a manual run from the
**Actions** tab if you want fresh numbers ahead of a demo.

## Install as a library

After the first release (`v1.0.0` tag), the package is on PyPI:

```bash
pip install costsight                  # core: detectors + alerts + attribution
pip install "costsight[api]"           # + FastAPI / uvicorn (backend for the web app)
pip install "costsight[llm]"           # + anthropic SDK for AI explanations
pip install "costsight[figures]"       # + matplotlib for slide figures
pip install "costsight[dev]"           # everything, plus pytest
```

Shell commands installed alongside the package:

```bash
costsight-pipeline --days 90 --seed 42 --scenario drift_heavy
costsight-benchmark --seeds 25
costsight-api --host 0.0.0.0 --port 8000
```

Programmatic use:

```python
from cloud_anomaly.synthetic_data import generate
from cloud_anomaly.detectors import DETECTORS
from cloud_anomaly.alerts import build_alerts
from cloud_anomaly.carbon import carbon_footprint

cur, labels, _ = generate(n_days=90, seed=42)
detections = DETECTORS["stl"](cur.groupby(["date","service"]).sum().reset_index())
alerts = build_alerts(detections, detector_name="stl", dataset_days=90)
carbon = carbon_footprint(cur)
print(f"This run emitted {carbon.kg_co2:.0f} kgCO₂-eq ({carbon.km_driven_equiv:.0f} km equiv).")
```

Releases are tag-driven: pushing `v1.x.y` triggers the
`.github/workflows/release.yml` workflow which builds the
sdist + wheel and publishes to PyPI via trusted-publishing
(no API token in the repo).

## Provision the cloud architecture

The production-path architecture documented in
[REPORT.md § 4.1](REPORT.md#cloud-architecture-production-path) is
shipped as a real Terraform module under [`terraform/`](terraform/):

```bash
cd terraform/
terraform init
terraform plan -var="env=dev" -var="alert_email=you@example.com"
terraform apply -var="env=dev" -var="alert_email=you@example.com"
```

Brings up: S3 raw + aggregated buckets, DynamoDB alerts table (PITR +
TTL), SNS alerts topic with optional email subscription, ingest Lambda
+ S3 trigger, and (optionally) a self-hosted dashboard ECS service.
Steady-state cost ~$5/mo per tenant at the default toggles.

## Scope

Phase 1 (May 20 deadline): synthetic data **and real AWS CUR ingestion**,
three detectors plus an ensemble vote, alert module, root-cause attribution,
P/R evaluation, multi-seed benchmark, a **19-view React web app** (summary /
forecast / carbon / drift / recommendations / tagging / AI-explain / lab /
replay / …) over a FastAPI backend, and statistical significance tests.
Phase 2 (post-finals): 3D/animation layer, comparison report extension,
paper-style writeup. Out of scope: real-time streaming and *automated*
multi-cloud ingestion — GCP/Azure billing is covered as a documented schema
mapping ([REPORT.md § 4.2](REPORT.md)), not a live adapter; only AWS CUR is
wired end-to-end. Production deployment of the detection pipeline stays batch
(the web app and REST API are deployable; the pipeline is not a streaming
service).

## License

[MIT](LICENSE) — see also [CONTRIBUTING.md](CONTRIBUTING.md) for how to
extend the project with new detectors or anomaly types.

## Authors

- **Furkan Can Karafil** ([@Urthella](https://github.com/Urthella)) · 222010020013
- **Halil Utku Demirtaş** · 222010020054
