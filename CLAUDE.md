# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

**costsight** - Project 13, Cloud Computing (Spring 2025-2026). An end-to-end cloud-cost anomaly detector with severity scoring, root-cause attribution, forecasting, carbon translation, and a 19-view **React** web app (Vite + TypeScript + Tailwind + Plotly) served over a **FastAPI** backend. Two-person student project (Halil Utku Demirtaş + Furkan Can Karafil); Phase 1 demo deadline **2026-05-20**.

> The UI was migrated from Streamlit to React for performance and a 3D/animation roadmap. The original Streamlit app is archived under [`legacy/`](legacy/) and tagged `streamlit-v1`; do not extend it.

The repo deliberately overshoots the original Phase 1 scope and now ships three distribution channels: source (this repo), PyPI (`pip install costsight` after a `v*.*.*` tag), and Terraform (`terraform/` module that stands up the full AWS architecture).

## Commands

The CI workflow (`.github/workflows/ci.yml`) is the canonical command list. Local dev uses PowerShell on Windows.

```powershell
# Install
pip install -r requirements.txt

# Full pipeline (writes data/raw/ + outputs/)
python scripts/run_pipeline.py

# Multi-seed Monte Carlo benchmark (writes outputs/benchmark_*.csv)
python scripts/run_benchmark.py --seeds 25

# Regenerate the 4 presentation PNGs under slides/figures/
python scripts/make_figures.py

# REST API - the frontend's data source (port 8000, OpenAPI at /docs)
uvicorn cloud_anomaly.api:app --reload --port 8000

# Tests
pytest -q                              # all
pytest -q tests/test_pipeline.py::test_each_detector_runs   # single test
```

The web UI lives in `frontend/` (Vite + React + TS). It calls the FastAPI
backend, so run the API first, then:

```powershell
cd frontend
npm install              # first time only
npm run dev              # dev server on :5173, proxies /api -> :8000
npm run build            # tsc + vite production build into frontend/dist
```

After a successful `pip install -e .` (or `pip install costsight` once released), three console scripts become available: `costsight-pipeline`, `costsight-benchmark`, `costsight-api`.

## Frontend notes (React + Vite)

- **One snapshot, many views.** The whole UI is driven by a single `GET /api/snapshot?scenario&n_days&seed` call (`build_snapshot` in `api.py`) cached by TanStack Query (`useSnapshot`). Adding a field to the snapshot is the way to feed a new view - don't add per-view endpoints unless the work is heavy/lazy (see `/api/perf`, `/api/explain`).
- **Plotly via factory.** `src/lib/plot.tsx` builds the component off `plotly.js-dist-min` through `react-plotly.js/factory`. Both are CommonJS, so the imports are unwrapped with `.default ?? mod` - without that you get `createPlotlyComponent is not a function` and a blank screen.
- **Routing = nav keys.** `src/nav.ts` is the source of truth (5 groups / 19 views, short ASCII keys + Material/Lucide icons); `App.tsx` maps each key to a view component. Theme tokens live in `src/index.css` (`@theme`, Tailwind v4).

## Architecture

### Detector contract (the single most important convention)

Every detector - Z-Score, STL, Isolation Forest, and Ensemble (consensus vote) - exposes the same `detect(long_df) -> pd.DataFrame` signature and returns the same schema:

```
date, service, cost, score, is_anomaly
```

Detectors are registered in `src/cloud_anomaly/detectors/__init__.py::DETECTORS`. Because every layer downstream (alerts, attribution, evaluation, dashboard tabs) keys off this schema, **adding a new detector is a strict extension** - drop a new module with `detect()`, register it, and every comparison view picks it up automatically.

### Data flow

```
synthetic_data.generate(scenario=...)           # or cur_loader.load_cur_csv() for real AWS CUR
  → long format: date, service, region, usage_type, cost, tag_team, tag_environment
  → preprocessing.aggregate_by_service / aggregate_daily
  → detectors/*.detect()                        # 4 implementations, common schema
  → alerts.build_alerts()                       # severity = deviation × duration × $impact
  → attribution.attribute()                     # picks dominant (region|usage_type|tag_*) value
  → evaluation.compare_detectors() / time_to_detect / cost_saved_estimate
  → forecast.forecast_per_service() (Holt-Winters, 90% PI)
  → drift.detect_drift() (Page-Hinkley + ADWIN baseline shift)
  → carbon.carbon_footprint() (USD → kgCO₂-eq via per-service kWh × per-region CO₂/kWh)
  → clustering.cluster_alerts() (DBSCAN → incidents)
  → explainer.explain_alert() (Claude API; falls back to deterministic template)
```

`pipeline.run()` is the orchestrator script - it writes detections / alerts / attribution / comparison / benchmark CSVs into `outputs/`. The React frontend (`frontend/`) and the FastAPI app (`src/cloud_anomaly/api.py`) are the live path: the API calls into these modules and the frontend renders the result; the archived Streamlit app (`legacy/`) called the same modules directly. They are alternative *views*, not parallel implementations.

### Scenario presets

`synthetic_data.SCENARIOS` exposes seven anomaly mixes: `default`, `drift_heavy`, `spike_storm`, `stealth_leak`, `multi_region`, `weekend_camouflage`, `calm`. The **`default` scenario preserves the original anomaly mix** so the committed 25-seed benchmark numbers in `outputs/benchmark_*.csv` and `REPORT.md` remain reproducible. Do not change `_scenario_anomalies("default", ...)` without rerunning the benchmark.

### LLM explainer

`src/cloud_anomaly/explainer.py` calls the Anthropic SDK when `ANTHROPIC_API_KEY` is set; otherwise it emits a deterministic templated explanation with the same shape so the dashboard layout never depends on a key being present. Default model is `claude-haiku-4-5` (cheapest fast model); cap is `max_tokens=400`; results are cached per `(alert_date, service, severity)` so dashboard reruns don't reburn tokens.

### Carbon translation

`src/cloud_anomaly/carbon.py` ships an inline snapshot of (a) per-service energy intensity in kWh/$ and (b) per-region grid carbon intensity in kgCO₂/kWh. Dated 2024-12-15. **Do not introduce a live API call here** - the dashboard renders the Carbon tab on every refresh and a per-render network call would be wasteful. The user-triggered `pricing.fetch_live()` in `pricing.py` is the pattern for opt-in live fetches.

## Repo conventions

- **Commit messages: NO AI co-author trailers.** The user's global `~/.claude/CLAUDE.md` forbids `Co-Authored-By: Claude …` and `Generated with Claude Code` footers. Just write the message.
- **PowerShell-first.** Windows is the dev environment. Use `Get-NetTCPConnection` / `Stop-Process`, not bash. `Bash` tool works via Git Bash but PowerShell is preferred for OS-level ops.
- **Cache server work, not the client.** Heavy module calls are bundled into `build_snapshot` and cached client-side by TanStack Query (keyed on scenario/n_days/seed). Don't add a new round-trip per view - extend the snapshot.
- **No comments on the WHAT.** Comments are reserved for non-obvious WHY (e.g. "preserves the original benchmark numbers" in `_scenario_anomalies`). Don't write multi-paragraph docstrings.
- **Tests live in `tests/test_pipeline.py`.** Every new module adds at least one smoke test there. Tests assume the working directory is the repo root (they `sys.path.insert(0, str(ROOT / "src"))`).

## Cross-references

- [`README.md`](README.md) - quick-start, install-as-library, Docker, frontend build + API deploy steps.
- [`REPORT.md`](REPORT.md) - full technical report. § 3.5 (statistical significance), § 4.1 (cloud architecture diagram), § 4.2 (multi-cloud schema mapping).
- [`DEMO.md`](DEMO.md) - 2.5-minute timed demo-video script.
- [`PRESENTATION.md`](PRESENTATION.md) - presentation-day map: slide↔demo timeline, role split, Q&A bank, pre-flight checklist, fallback plan.
- [`slides/deck.md`](slides/deck.md) - Marp slide deck; `slides/deck.html` is the current rendered build. Regenerate a PDF with `npx @marp-team/marp-cli slides/deck.md --pdf -o slides/deck.pdf` (needs a local Chrome/Edge).
- [`terraform/README.md`](terraform/README.md) - how to `terraform init/plan/apply` the production architecture into AWS.
- [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb) - end-to-end notebook reproducing every view in static form.
- [`frontend/`](frontend/) - React web app (Vite + TS + Tailwind + Plotly); `legacy/` holds the archived Streamlit app.
