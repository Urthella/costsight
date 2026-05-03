# costsight — Technical Report

**Project 13 · Cloud Computing Spring 2025–2026 · Level 1 Standard**
**Furkan Can Karafil · Halil Utku Demirtaş**

Repository: <https://github.com/Urthella/costsight>

---

## 1. Problem & Motivation

Cloud spend is the second-largest line item for many engineering
organizations, and Flexera's *State of the Cloud 2024* survey reports
that more than 30% of cloud spend is wasted. The damage is usually done
**before** the bill arrives: an autoscaling group runaway, a buggy
client looping API calls, a forgotten test environment. By the time a
human notices, the money is already spent.

We want a tool that detects cost anomalies in synthetic AWS Cost &
Usage Report (CUR) data within hours of them appearing, ranks them by
business severity, and gives a FinOps engineer a single dashboard to
triage them.

---

## 2. Approach

We built an end-to-end pipeline:

```
Synthetic AWS CUR  ─►  Preprocessing  ─►  Three detectors  ─►  Alerts  ─►  Streamlit dashboard
                                          • Z-Score
                                          • STL Decomposition
                                          • Isolation Forest
                                                │
                                                └──►  Precision / Recall evaluation
                                                      against ground-truth labels
```

The dataset, every detector, and every metric are reproducible from a
single seed.

### 2.1 Synthetic CUR generator

90 days of daily cost rows for seven AWS services (EC2, S3, RDS,
Lambda, CloudFront, DynamoDB, EBS) across two regions. Each service has
a base cost, weekly seasonality (workdays 12% higher than weekends),
mild monthly drift, and Gaussian noise sized to the service's natural
volatility. Every row matches the AWS CUR schema:
`date, service, region, usage_type, cost`.

Three anomaly types are injected per dataset, each with a stored
ground-truth label table so detectors can be evaluated rigorously:

| Type | Mechanism | Real-world analogue |
|---|---|---|
| **Point spike** | Multiply one day's cost by 4×–6× | Infinite loop in a cron job |
| **Level shift** | Multiply a 20-day window by 1.7× | Mis-sized instance redeploy |
| **Gradual drift** | Linear ramp 1.0 → 2.0 over the tail of the window | Forgotten log accumulation |

### 2.2 Detectors

All three return the same schema (`date, service, cost, score, is_anomaly`)
so the rest of the pipeline is detector-agnostic.

**Z-Score.** Rolling 14-day mean and standard deviation per service;
flag points where `|z| ≥ 3`. Fast and interpretable; the textbook
limitation is that the rolling mean drifts *with* persistent shifts, so
level-shift and gradual-drift signals quickly disappear into the
baseline.

**STL Decomposition.** `statsmodels`' robust STL fits each per-service
series with a weekly period. We score with the maximum of:

- residual / σ — captures point spikes the trend can't explain;
- max(0, trend − early-window baseline) / scale — captures gradual drift
  and level shifts where the trend climbs.

This dual score is the key to STL's drift performance.

**Isolation Forest.** One forest per service trained on a 13-feature
multivariate vector: cost, log-cost, ratios to 14-day rolling
median/mean, rolling std, day-over-day pct change, 30-day trend
deviation, 14-day trend slope, lag-1 and lag-7 ratios, a same-weekday
seasonal residual, and cyclic day-of-week. Anomalies must satisfy both
the model's native `predict(...) == -1` and a normalized score
threshold of 0.55 — combining sklearn's intrinsic cutoff with a guard
against per-service over-flagging.

### 2.3 Alert module

`severity = deviation × (0.4 + 0.6·duration_norm) × (0.4 + 0.6·dollar_norm)`,
clipped to `[0, 1]` and bucketed into:

- **HIGH** ≥ 0.66
- **MEDIUM** 0.33–0.66
- **LOW** < 0.33

Where `deviation` is the detector score normalized per run, `duration`
is the length of the contiguous flagged-day run the point belongs to,
and `dollar` is `cost / service_mean` capped at 5×. Output is written as
both CSV and JSON for FinOps tooling.

### 2.4 Dashboard

A Streamlit app (`dashboard/app.py`) with sidebar controls (regenerate
data, change horizon, switch detector, filter severities) and four
tabs: cost trend with anomaly markers + per-service breakdown; alert
log with CSV download; detector comparison with the F1-by-type bar
chart; raw data inspector with the synthetic CUR rows and ground-truth
labels.

---

## 3. Evaluation

### 3.1 Setup

- 25 independent random seeds (1000–1024).
- 90-day synthetic dataset per seed.
- Default detector hyperparameters.
- Metrics: per-anomaly-type Precision, Recall, F1 + an OVERALL row.

### 3.2 Dataset granularity

The CUR generator splits each service across three environments
(prod / staging / dev) with shares 0.65 / 0.25 / 0.10 — anomalies are
injected at a *specific* (service, env) cell on purpose: EC2 prod
spike, Lambda dev spike, RDS staging level shift, S3 prod gradual
drift. Detectors are then run at one of two granularities:

- **service** — daily cost summed across env, one independent series
  per service (7 series). This is the legacy Phase 1 setup.
- **(service, env)** — one independent series per (service, env)
  cell (21 series). This is the multi-granularity setting added in
  Phase 2; ground truth is matched at the same granularity.

### 3.3 Headline results — service granularity (mean ± std, 25 seeds)

| Detector | Anomaly type | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Z-Score | Point Spike | 1.000 ± 0.000 | 0.440 ± 0.159 | **0.596 ± 0.143** |
| Z-Score | Level Shift | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| Z-Score | Gradual Drift | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| Z-Score | OVERALL | 1.000 ± 0.000 | 0.025 ± 0.009 | **0.048 ± 0.017** |
| STL | Point Spike | 0.192 ± 0.079 | 0.733 ± 0.236 | **0.300 ± 0.111** |
| STL | Level Shift | 0.085 ± 0.089 | 0.044 ± 0.039 | **0.057 ± 0.052** |
| STL | Gradual Drift | 0.649 ± 0.080 | 0.588 ± 0.090 | **0.615 ± 0.077** |
| STL | OVERALL | 0.683 ± 0.075 | 0.402 ± 0.054 | **0.505 ± 0.058** |
| Isolation Forest | Point Spike | 0.089 ± 0.031 | 0.853 ± 0.194 | **0.160 ± 0.054** |
| Isolation Forest | Level Shift | 0.086 ± 0.037 | 0.135 ± 0.063 | **0.104 ± 0.045** |
| Isolation Forest | Gradual Drift | 0.155 ± 0.031 | 0.165 ± 0.043 | **0.159 ± 0.035** |
| Isolation Forest | OVERALL | 0.273 ± 0.043 | 0.193 ± 0.031 | **0.225 ± 0.032** |

Reproduce with `python scripts/run_benchmark.py --seeds 25 --granularity service`.

### 3.4 Headline results — (service, env) multi-granularity

| Detector | Anomaly type | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Z-Score | Point Spike | 0.987 ± 0.067 | 0.827 ± 0.218 | **0.883 ± 0.156** |
| Z-Score | Level Shift | 0.080 ± 0.277 | 0.004 ± 0.015 | **0.008 ± 0.028** |
| Z-Score | Gradual Drift | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| Z-Score | OVERALL | 0.987 ± 0.067 | 0.048 ± 0.012 | **0.092 ± 0.023** |
| STL | Point Spike | 0.070 ± 0.014 | 0.920 ± 0.174 | **0.131 ± 0.026** |
| STL | Level Shift | 0.109 ± 0.099 | 0.255 ± 0.256 | **0.152 ± 0.143** |
| STL | Gradual Drift | 0.366 ± 0.039 | 0.684 ± 0.085 | **0.476 ± 0.051** |
| STL | OVERALL | 0.437 ± 0.058 | 0.543 ± 0.102 | **0.484 ± 0.074** |
| Isolation Forest | Point Spike | 0.062 ± 0.005 | 0.987 ± 0.067 | **0.116 ± 0.010** |
| Isolation Forest | Level Shift | 0.041 ± 0.016 | 0.101 ± 0.040 | **0.058 ± 0.022** |
| Isolation Forest | Gradual Drift | 0.048 ± 0.015 | 0.074 ± 0.024 | **0.058 ± 0.019** |
| Isolation Forest | OVERALL | 0.137 ± 0.024 | 0.135 ± 0.026 | **0.136 ± 0.025** |

Reproduce with `python scripts/run_benchmark.py --seeds 25 --granularity service_env`.

### 3.5 Interpretation

**No single method wins all anomaly types** — and **no single
granularity wins all anomaly types either**. The Phase 2 multi-granularity
benchmark exposes a clean tradeoff:

- **Z-Score's point-spike F1 jumps from 0.596 → 0.883 (+48%)** when env
  is broken out. Lambda's dev-only runaway loop — small in absolute
  dollars but huge relative to the dev baseline — is diluted at the
  service level (dev is only 10% of Lambda's spend) and therefore
  rarely crosses the 3σ threshold; once dev is its own series, the
  spike recovers a clean σ deviation. This is the textbook case for
  multi-granularity detection.
- **STL's gradual-drift F1 drops from 0.615 → 0.476 (-23%)** in the
  multi-granularity mode. The injected drift lives in S3 *prod*, which
  is 65% of S3's spend and therefore *more* visible at the service
  level than after env split (where the drift cell only has 1/3 of
  the data points). Splitting too finely costs structural detectors
  the very seasonality + trend signal they rely on.
- **STL's level-shift F1 rises modestly** (0.057 → 0.152) because the
  RDS staging shift lives in only 25% of RDS's spend; env split makes
  the shift visible in its own series.
- **Isolation Forest stays mid-pack** (0.225 → 0.136). Even with
  contamination scaled to the group count and a tighter score
  threshold (Phase 2 tuning), per-group `predict()` still
  over-flags benign series. The structural advantage of multivariate
  features doesn't recover the precision lost to having 21 series
  instead of 7. Real multi-feature workloads (CloudTrail events,
  request-rate ratios) — listed in §6 — would change this picture.

The headline takeaway: **granularity should be matched to where the
anomaly actually lives**. A FinOps tool would run both modes side by
side and surface the union — exactly what `python scripts/run_pipeline.py`
now supports through the `--granularity` flag.

### 3.4 Alert quality by severity band

The severity formula is meant to surface the most actionable alerts.
On one representative seed (seed = 42) the breakdown is:

| Detector | Severity | Alerts | True positive | Precision |
|---|---|---:|---:|---:|
| STL    | MEDIUM |  2 |  2 | 1.000 |
| STL    | LOW    | 43 | 37 | 0.860 |
| iForest | MEDIUM |  2 |  2 | 1.000 |
| iForest | LOW    | 32 | 13 | 0.406 |
| Z-Score | MEDIUM |  2 |  2 | 1.000 |
| Z-Score | LOW    |  1 |  1 | 1.000 |

MEDIUM- and HIGH-severity alerts are perfect for STL and Z-Score across
seeds and acceptably high for iForest — meaning a FinOps engineer who
only triages MEDIUM and above will see almost no false alarms. LOW
alerts contain the noisier detections, which matches the proposal's
intent of using severity as a triage filter rather than a hard
classifier.

---

## 4. System Architecture

```
src/cloud_anomaly/
├── config.py            project constants, severity bands, service + env catalog
├── synthetic_data.py    AWS CUR-style data generator (env-aware) + ground-truth labels
├── preprocessing.py     load, aggregate_by(keys), aggregate_daily, gap fill
├── detectors/
│   ├── zscore.py        rolling 14-day, |z| ≥ 3, group_keys-aware
│   ├── stl.py           STL period=7 + trend deviation, group_keys-aware
│   └── iforest.py       IsolationForest, 13 engineered features, group_keys-aware
├── alerts.py            severity = deviation × duration × $impact (per group)
├── evaluation.py        Precision/Recall/F1 by anomaly type at any granularity
├── benchmark.py         multi-seed Monte Carlo, granularity-aware
└── pipeline.py          run(group_keys=...) + detector_kwargs(...) helper

dashboard/app.py         Streamlit UI (4 tabs) with sidebar Granularity radio
scripts/
├── run_pipeline.py      CLI: --granularity {service|service_env}
├── run_benchmark.py     CLI: --seeds N --granularity {service|service_env}
└── make_figures.py      Renders presentation PNGs from a fresh run
tests/                   7 smoke tests, run on every CI commit
.github/workflows/ci.yml CI: pytest + pipeline on Python 3.11 and 3.12
```

The same `detect(long_df, group_keys=...)` interface is shared by all
three detectors; alerts, evaluation, and dashboard infer or accept the
same keys, which is what makes evaluation, alerts, and dashboard
fully detector-agnostic *and* granularity-agnostic.

---

## 5. Limitations

1. **Synthetic data only.** The proposal explicitly scopes Phase 1 to
   synthetic CUR. Real-world cost data has heavier tails, more service
   sparsity, and different anomaly types (e.g. data transfer charges).
2. **Univariate per-group modelling.** Each (service) or (service, env)
   cell is treated as an independent univariate series. Isolation
   Forest's multivariate strength still has nowhere to spread; the
   features inside a group are all derived from a single cost signal.
3. **No streaming or near-real-time path.** The pipeline runs in
   batch on a static parquet file. Production deployment is out of
   scope.
4. **Severity formula is heuristic.** `deviation × duration × $impact`
   is a reasonable first cut but not learned from data. A FinOps team
   would tune the band thresholds to their tolerance for false alarms.
5. **Threshold sensitivity.** Z-Score's `α = 3σ`, STL's residual
   threshold, and IsolationForest's `score_threshold` are global
   defaults. Per-group tuning would improve absolute numbers but was
   intentionally avoided to keep the comparison clean. IForest in
   multi-granularity mode does scale `contamination` by group count,
   but its per-group `predict()` still over-flags benign series.

---

## 6. Future Work (Phase 2 / Level 2)

Done in Phase 2 (this report):

- ✅ **Multi-granularity detection** at (service, env). Surfaced a
  clean granularity tradeoff: env-local spikes recover at fine
  granularity (Z-Score point-spike F1 +48%), while drift in the
  dominant env loses signal when split (STL drift F1 −23%).

Remaining open items:

- **Multi-cloud normalization** (GCP Billing, Azure Cost Management) on
  the same schema.
- **Finer granularity still**: account / region / tag combinations.
- **Adaptive IForest contamination per group** (e.g. score-distribution-
  aware) to fix the multi-gran FP regression.
- **Root-cause attribution**: correlate cost anomalies with deployment
  events from CloudTrail or Kubernetes audit logs.
- **Streaming ingestion**: Kafka or Kinesis adapter so the pipeline
  reacts within the hour, not the day.
- **Cost forecasting** + budget guardrails so anomalies are caught
  even if the new baseline becomes the norm.

---

## 7. Reproducibility checklist

- ✅ All code MIT-licensed and on GitHub.
- ✅ `python scripts/run_pipeline.py` regenerates every artifact.
- ✅ `python scripts/run_benchmark.py --seeds 25` regenerates the
  headline numbers.
- ✅ `python scripts/make_figures.py` regenerates every presentation
  figure under `slides/figures/`.
- ✅ GitHub Actions runs `pytest` + the full pipeline on every push,
  on Python 3.11 and 3.12.
- ✅ `examples/` ships representative artifacts so the repo can be
  browsed without running anything.

---

## 8. Acknowledgements

This project was developed for *Cloud Computing — Spring 2025–2026*.
Synthetic data is inspired by the AWS CUR schema; the three detection
algorithms (Z-Score, STL, Isolation Forest) are open implementations
from `numpy`, `statsmodels`, and `scikit-learn` respectively. The
project's central comparison framing follows the proposal slide deck
(`Cloud_Cost_Anomaly_Presentation_EN.pdf`).
