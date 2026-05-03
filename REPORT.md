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

### 3.2 Headline results (mean ± std across 25 seeds)

| Detector | Anomaly type | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Z-Score | Point Spike | 0.990 ± 0.050 | 0.947 ± 0.125 | **0.962 ± 0.078** |
| Z-Score | Level Shift | 0.120 ± 0.332 | 0.006 ± 0.017 | **0.012 ± 0.033** |
| Z-Score | Gradual Drift | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** |
| Z-Score | OVERALL | 0.990 ± 0.050 | 0.056 ± 0.010 | **0.105 ± 0.018** |
| STL | Point Spike | 0.357 ± 0.078 | 1.000 ± 0.000 | **0.522 ± 0.082** |
| STL | Level Shift | 0.640 ± 0.179 | 0.611 ± 0.236 | **0.616 ± 0.204** |
| STL | Gradual Drift | 0.789 ± 0.057 | 0.689 ± 0.065 | **0.734 ± 0.052** |
| STL | OVERALL | 0.862 ± 0.043 | 0.678 ± 0.085 | **0.757 ± 0.064** |
| Isolation Forest | Point Spike | 0.141 ± 0.023 | 1.000 ± 0.000 | **0.247 ± 0.035** |
| Isolation Forest | Level Shift | 0.198 ± 0.056 | 0.240 ± 0.070 | **0.216 ± 0.060** |
| Isolation Forest | Gradual Drift | 0.246 ± 0.037 | 0.196 ± 0.035 | **0.217 ± 0.034** |
| Isolation Forest | OVERALL | 0.424 ± 0.048 | 0.257 ± 0.034 | **0.319 ± 0.036** |

Reproduce with `python scripts/run_benchmark.py --seeds 25`.

### 3.3 Interpretation

**No single method wins all anomaly types** — the central thesis of the
project is empirically supported. The takeaways:

- **STL leads overall** at F1 = 0.757 ± 0.064. Decomposing trend +
  seasonality + residual gives it eyes on every anomaly type. Drift
  performance (F1 = 0.734) is the strongest of the three detectors.
- **Z-Score is the sharpest point-spike detector** (F1 = 0.962, near
  perfect at α = 3σ), but a stationary baseline simply cannot detect
  level shifts or drift (F1 ≈ 0). This is exactly what a stats text
  predicts and validates that the proposal's qualitative ratings
  matched reality.
- **Isolation Forest is mid-pack** (F1 = 0.319). It catches every point
  spike (recall = 1.0 there) and partially recovers level-shift /
  drift signal once we engineer lag, slope, and seasonal-residual
  features, but still cannot match STL's structural decomposition on
  univariate cost. Its strength would shine on multi-cloud, multi-tag,
  multi-feature workloads — out of scope here, listed as future work.

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
├── config.py            project constants, severity bands, service catalog
├── synthetic_data.py    AWS CUR-style data generator + ground-truth labels
├── preprocessing.py     load, aggregate (daily/per-service), gap fill
├── detectors/
│   ├── zscore.py        rolling 14-day, |z| ≥ 3
│   ├── stl.py           STL period=7 + trend deviation
│   └── iforest.py       IsolationForest, 13 engineered features
├── alerts.py            severity = deviation × duration × $impact
├── evaluation.py        Precision/Recall/F1 by anomaly type + alert quality
├── benchmark.py         multi-seed Monte Carlo
└── pipeline.py          run() — wires everything together

dashboard/app.py         Streamlit UI (4 tabs)
scripts/
├── run_pipeline.py      CLI: full pipeline → outputs/
├── run_benchmark.py     CLI: 25-seed benchmark → outputs/benchmark_*.csv
└── make_figures.py      Renders presentation PNGs from a fresh run
tests/                   smoke tests, run on every CI commit
.github/workflows/ci.yml CI: pytest + pipeline on Python 3.11 and 3.12
```

The same `detect(df)` interface is shared by all three detectors, which
is what makes evaluation, alerts, and dashboard fully detector-agnostic.

---

## 5. Limitations

1. **Synthetic data only.** The proposal explicitly scopes Phase 1 to
   synthetic CUR. Real-world cost data has heavier tails, more service
   sparsity, and different anomaly types (e.g. data transfer charges).
2. **Univariate per-service modelling.** The pipeline treats each
   service's daily cost as an independent series. This penalizes
   Isolation Forest, whose strength is multi-feature anomaly density.
3. **No streaming or near-real-time path.** The pipeline runs in
   batch on a static parquet file. Production deployment is out of
   scope.
4. **Severity formula is heuristic.** `deviation × duration × $impact`
   is a reasonable first cut but not learned from data. A FinOps team
   would tune the band thresholds to their tolerance for false alarms.
5. **Threshold sensitivity.** Z-Score's `α = 3σ`, STL's residual
   threshold, and IsolationForest's `score_threshold` are global
   defaults. Per-service tuning would improve absolute numbers but was
   intentionally avoided to keep the comparison clean.

---

## 6. Future Work (Phase 2 / Level 2)

- **Multi-cloud normalization** (GCP Billing, Azure Cost Management) on
  the same schema.
- **Multi-granularity detection**: account / service / region / tag.
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
