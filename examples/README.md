# Example artifacts

These files are real outputs from one run of `python scripts/run_pipeline.py`
on the default 90-day synthetic dataset (`seed=42`). They are committed so
the repo can be browsed without first running the pipeline.

| File | What it is |
|---|---|
| `comparison.csv` | Per-anomaly-type Precision / Recall / F1 for all three detectors |
| `alerts_stl_sample.json` | Severity-banded alert log produced by the STL detector |

To regenerate everything:

```bash
python scripts/run_pipeline.py
```

Outputs land under `outputs/` (which is gitignored).
