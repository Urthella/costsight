# Demo Walkthrough — costsight

A timed script for the live/recorded demo (~2.5 minutes), built around the
**React web app** (Vite + Plotly + React Three Fiber) talking to the
**FastAPI** backend. It hits every Phase 1 deliverable in the order a reviewer
expects, and leans on the guided tour + 3D so the system shows itself off.

## Setup (off-camera)

```powershell
# Terminal 1 — backend (data source for the UI)
uvicorn cloud_anomaly.api:app --port 8000      # warms the default snapshot on boot

# Terminal 2 — web app
cd frontend
npm run dev                                     # http://localhost:5173
```

Or the whole stack in one shot:

```powershell
docker compose up --build                       # web on :8080, API on :8000
```

Open **http://localhost:5173** and let it load once (the default snapshot is
pre-warmed, so it appears in ~0.2 s). Set browser zoom to ~110% for the
recording. To replay the opening tour, click **Tour** (top-right) or run
`localStorage.clear()` and refresh.

## Recording script

| Time | Scene | Narration |
|---|---|---|
| **0:00 – 0:15** | App opens; the **guided tour** auto-plays (or click **Tour**) | "This is **costsight**, an automated cloud-cost anomaly detector for Project 13. The app introduces itself — a quick tour of the navigation, the live KPIs, the 3D charts, and the CSV upload." |
| **0:15 – 0:35** | **Summary** view. KPIs count up; orbit the 3D **spend skyline** | "The header KPIs animate in: total spend, anomalies flagged, consensus alerts, and dollars saveable. The 3D skyline is each service's spend — drag to orbit, hover a tower for its alert count. Below: distinct anomalies, the carbon footprint, and the single biggest savings action." |
| **0:35 – 0:55** | **3D explorer** | "This is the showcase. A WebGL spend skyline, a 3D cost *surface* where spikes become peaks and drift becomes rising ridges, and a 3D anomaly cloud — every (service, day) point, anomalies in red — that you can rotate to inspect." |
| **0:55 – 1:15** | **Detector comparison** (3D F1 bars; flip the **3D｜2D** toggle once) | "The core thesis: no single detector wins everywhere. STL, Isolation Forest, Z-Score and an ensemble vote run side by side, scored against ground-truth labels across 25 seeds. Z-Score is perfect on point spikes but blind to drift; STL leads overall at 0.76 F1. Every 3D chart has a 2D toggle for precise reading." |
| **1:15 – 1:35** | **Alert log** → **Root-cause** | "Anomalies become severity-banded alerts — severity is deviation × duration × dollar impact. Root-cause decomposes each day's spend by region / usage-type / tag against a 14-day baseline and emits a plain-English driver, e.g. *'us-east-1 drove 100% of the increase.'*" |
| **1:35 – 1:55** | **Forecast** (3D ribbons) → **Carbon** (3D bars) | "Holt-Winters forecasts every service as a 3D ribbon with a 90% prediction interval in 2D. The carbon tab translates dollars to kgCO₂e per service and region — the sustainability angle." |
| **1:55 – 2:20** | **Upload AWS CUR** in the sidebar → drop `examples/cur_spike_storm_60d.csv` | "And it isn't tied to synthetic data — drop a real AWS Cost & Usage Report and **every view re-runs on it**: detection, alerts, forecast, carbon. This is the same tool a FinOps team would point at their own bill." |
| **2:20 – 2:30** | Cut to `pytest` pass + GitHub Actions CI badge | "Backend is 21 green tests behind a CI pipeline that also builds the frontend. Thanks for watching." |

## If the reviewer hands you a CSV

The sidebar **Upload AWS CUR (.csv)** widget accepts any AWS CUR export
(it auto-detects `lineItem/UsageStartDate`, `lineItem/ProductCode`,
`lineItem/UnblendedCost`, region, usage-type and tag columns). Ready-made
demo files live in [`examples/`](examples/): `cur_default_90d`,
`cur_spike_storm_60d`, `cur_stealth_leak_90d`, `cur_multi_region_90d`,
`cur_calm_60d`. Drop one and narrate what changes. (Detector comparison
goes blank for uploaded data — real bills ship no ground-truth labels;
say so, it's the honest answer.)

## Q&A talking points

- **Architecture:** Python backend (`src/cloud_anomaly/`, 24 modules) behind a
  FastAPI app; the React frontend fetches one cached `/api/snapshot` per
  (scenario, days, seed) and fans it out across 19 views. Clean API/UI split.
- **Detector contract:** every detector exposes `detect(long_df)` returning
  `date, service, cost, score, is_anomaly` — adding one is a strict extension.
- **Severity** = deviation × duration × dollar_impact → LOW / MEDIUM / HIGH.
- **Why synthetic:** injected anomalies give ground truth, which is how we get
  real Precision/Recall/F1 (mean ± std over 25 seeds).
- **Why 3D:** it's genuinely informative here (surfaces show drift, clouds show
  outliers) and every 3D view has a 2D toggle; animation honors
  `prefers-reduced-motion`.
- **Performance:** server-side snapshot caching + startup warm-up (~0.2 s loads),
  frontend code-splitting (three.js loads only for 3D views).

## What to show vs. skip

**Show:** the tour, the Summary skyline, the 3D explorer, detector comparison
(3D + the 2D toggle), alert log + root-cause, and the **CUR upload**. These map
to the deliverable list and land the 3D "wow".

**Skip on camera (mention only if asked):** code internals, the Terraform/PyPI
channels, and the archived Streamlit app under `legacy/`.

## Recording tools

- Free capture: OBS Studio, or Windows Game Bar (`Win+G`).
- Quick edits: ClipChamp (built into Windows 11) — cuts + an intro/outro slide
  are plenty.
