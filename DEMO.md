# Demo Walkthrough — costsight

A timed script for the Phase 1 demo video (~2 minutes 10 seconds total).
Designed to hit every Phase 1 deliverable in roughly the order a reviewer
expects.

> Tip before recording: run `python scripts/run_pipeline.py` once to seed
> `data/raw/` and `outputs/`, then start the dashboard and let it warm up
> before hitting record.

## Setup (off-camera)

```powershell
# Terminal 1 — pipeline + outputs
python scripts/run_pipeline.py

# Terminal 2 — dashboard
streamlit run dashboard/app.py
```

Both should be running before you start recording. Set the browser zoom
to 110% so the dashboard text is readable in the recording.

## Recording script

| Time | Scene | Narration |
|---|---|---|
| **0:00 – 0:10** | Title slide / repo page | "This is **costsight**, an automated cloud cost anomaly detector built on AWS Cost & Usage Report data, for Project 13 in Cloud Computing. The full source is on GitHub — `Urthella/costsight`." |
| **0:10 – 0:25** | Open the dashboard's **Cost trend** tab. Hover the daily-cost chart | "The dashboard ingests 90 days of synthetic CUR data covering seven AWS services. Each red marker is a day flagged by our active detector — by default STL Decomposition." |
| **0:25 – 0:40** | Switch the *Active detector* dropdown in the sidebar to **Z-Score**, watch the markers update | "Switching to the Z-Score baseline shows how a single algorithm behaves on the same data — it nails the point spike but misses the persistent shift in RDS spend, exactly as the literature predicts." |
| **0:40 – 0:55** | Switch back to **STL Decomposition** | "STL splits each service's series into trend, seasonal, and residual components. That gives it eyes on both sudden spikes and the gradual S3 drift starting around day 60." |
| **0:55 – 1:10** | Open the **Alert log** tab. Show severity bands, download CSV | "Anomalies become alerts. Severity is computed as deviation × duration × dollar impact, then bucketed into LOW, MEDIUM, HIGH so a FinOps team can triage. Every alert is exportable as JSON or CSV." |
| **1:10 – 1:25** | Open the **Root-cause** tab. Point at one summary line | "For every alert we decompose the day's spend by region and usage-type, compare against a 14-day baseline, and emit a one-line hint — for example, *'us-east-1 region drove 100% of the increase.'* That's what turns a flag into something a FinOps engineer can act on." |
| **1:25 – 1:50** | Open **Detector comparison** tab. Show the F1-by-type bar chart | "We don't just run one algorithm — STL, Isolation Forest, and Z-Score run side by side, evaluated against ground-truth anomaly labels generated alongside the dataset. Z-Score is perfect on point spikes but blind to drift; STL leads overall at 0.76 F1 across 25 seeds; Isolation Forest catches multi-feature anomalies the others miss. The point of the project is exactly this comparison — no single method wins all anomaly types." |
| **1:50 – 2:05** | Open the **Raw data** tab briefly | "The dashboard also exposes the raw CUR rows and the ground-truth labels, so anyone can audit how a flagged day was scored." |
| **2:05 – 2:10** | Cut to a quick `pytest` pass + GitHub Actions CI badge | "Everything is reproducible — pytest and a CI pipeline gate every push. Thanks for watching." |

## Key talking points to weave in (if asked)

- "We use *synthetic* CUR data so we can inject anomalies and measure
  Precision/Recall — that's how we get the F1 numbers in the comparison."
- Severity formula matches the proposal slide:
  `severity = deviation × duration × dollar_impact`.
- Three injected anomaly types: point spike, level shift, gradual drift.
- Phase 1 scope is deliberately bounded: synthetic data, three detectors,
  alerts, dashboard, P/R. Real-time streaming, multi-cloud, and
  auto-remediation are explicitly out of scope.

## What to show vs. what to skip

**Show:** the cost-trend chart, detector switching, the alert log,
severity colors, the root-cause summary, the comparison F1 chart.
These map 1:1 to the deliverable list on the proposal.

**Skip on camera (mention only if asked):**
- Code internals — the demo is for the system, not the source.
- The notebooks folder (currently empty placeholder).
- Slow first-run model fits — pre-warm by running the pipeline once.

## Recording tools

- Free: OBS Studio (full-screen capture), Windows Game Bar (`Win+G`).
- Quick edits: ClipChamp (built into Windows 11) is enough for cuts +
  intro/outro slide. No need for fancy effects.
