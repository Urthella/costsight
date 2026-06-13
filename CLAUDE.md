# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

**costsight** — Project 13, Cloud Computing (Spring 2025–2026). An end-to-end cloud-cost anomaly detector with severity scoring, root-cause attribution, forecasting, carbon translation, and an 18-tab Streamlit dashboard. Two-person student project (Halil Utku Demirtaş + Furkan Can Karafil); Phase 1 demo deadline **2026-05-20**.

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

# Dashboard (port 8501)
python -m streamlit run dashboard/app.py --server.headless true --server.port 8501

# REST API (port 8000, OpenAPI at /docs)
uvicorn cloud_anomaly.api:app --reload --port 8000

# Tests
pytest -q                              # all
pytest -q tests/test_pipeline.py::test_each_detector_runs   # single test
```

After a successful `pip install -e .` (or `pip install costsight` once released), three console scripts become available: `costsight-pipeline`, `costsight-benchmark`, `costsight-api`.

## Streamlit gotchas

- **Module cache survives hot reload for `from X import Y` statements.** When you add a new name to a module that the dashboard imports, Streamlit's auto-reload often fails with `ImportError: cannot import name '...'`. Fix: kill the Streamlit process (`Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue` → `Stop-Process -Force`) and relaunch. Hit this twice in development — it is not a code bug.
- **`Regenerate synthetic data` checkbox + scenario dropdown share a session_state signature** (`dashboard/app.py`). Changing scenario / n_days / seed automatically forces a regenerate even when the checkbox is off — the user expects scenario switches to update the dataset, and `_load`'s `@st.cache_data` key was not enough on its own.

## Architecture

### Detector contract (the single most important convention)

Every detector — Z-Score, STL, Isolation Forest, and Ensemble (consensus vote) — exposes the same `detect(long_df) -> pd.DataFrame` signature and returns the same schema:

```
date, service, cost, score, is_anomaly
```

Detectors are registered in `src/cloud_anomaly/detectors/__init__.py::DETECTORS`. Because every layer downstream (alerts, attribution, evaluation, dashboard tabs) keys off this schema, **adding a new detector is a strict extension** — drop a new module with `detect()`, register it, and every comparison view picks it up automatically.

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

`pipeline.run()` is the orchestrator script — it writes detections / alerts / attribution / comparison / benchmark CSVs into `outputs/`. The dashboard (`dashboard/app.py`) and the FastAPI app (`src/cloud_anomaly/api.py`) call into the same modules; they are alternative *views*, not parallel implementations.

### Scenario presets

`synthetic_data.SCENARIOS` exposes seven anomaly mixes: `default`, `drift_heavy`, `spike_storm`, `stealth_leak`, `multi_region`, `weekend_camouflage`, `calm`. The **`default` scenario preserves the original anomaly mix** so the committed 25-seed benchmark numbers in `outputs/benchmark_*.csv` and `REPORT.md` remain reproducible. Do not change `_scenario_anomalies("default", ...)` without rerunning the benchmark.

### LLM explainer

`src/cloud_anomaly/explainer.py` calls the Anthropic SDK when `ANTHROPIC_API_KEY` is set; otherwise it emits a deterministic templated explanation with the same shape so the dashboard layout never depends on a key being present. Default model is `claude-haiku-4-5` (cheapest fast model); cap is `max_tokens=400`; results are cached per `(alert_date, service, severity)` so dashboard reruns don't reburn tokens.

### Carbon translation

`src/cloud_anomaly/carbon.py` ships an inline snapshot of (a) per-service energy intensity in kWh/$ and (b) per-region grid carbon intensity in kgCO₂/kWh. Dated 2024-12-15. **Do not introduce a live API call here** — the dashboard renders the Carbon tab on every refresh and a per-render network call would be wasteful. The user-triggered `pricing.fetch_live()` in `pricing.py` is the pattern for opt-in live fetches.

## Repo conventions

- **Commit messages: NO AI co-author trailers.** The user's global `~/.claude/CLAUDE.md` forbids `Co-Authored-By: Claude …` and `Generated with Claude Code` footers. Just write the message.
- **PowerShell-first.** Windows is the dev environment. Use `Get-NetTCPConnection` / `Stop-Process`, not bash. `Bash` tool works via Git Bash but PowerShell is preferred for OS-level ops.
- **`@st.cache_data` for any function that touches the disk or runs a detector.** The dashboard reruns the whole script on every interaction; without caching, detector latency makes the UI feel laggy.
- **No comments on the WHAT.** Comments are reserved for non-obvious WHY (e.g. "preserves the original benchmark numbers" in `_scenario_anomalies`). Don't write multi-paragraph docstrings.
- **Tests live in `tests/test_pipeline.py`.** Every new module adds at least one smoke test there. Tests assume the working directory is the repo root (they `sys.path.insert(0, str(ROOT / "src"))`).

## Cross-references

- [`README.md`](README.md) — quick-start, install-as-library, Docker, Streamlit Cloud deploy steps.
- [`REPORT.md`](REPORT.md) — full technical report. § 3.5 (statistical significance), § 4.1 (cloud architecture diagram), § 4.2 (multi-cloud schema mapping).
- [`DEMO.md`](DEMO.md) — 2-minute timed demo-video script.
- [`slides/deck.md`](slides/deck.md) — Marp slide deck; `slides/deck.pdf` is the rendered build.
- [`terraform/README.md`](terraform/README.md) — how to `terraform init/plan/apply` the production architecture into AWS.
- [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb) — end-to-end notebook reproducing every dashboard tab in static form.
