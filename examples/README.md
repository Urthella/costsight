# Example artifacts

These files are real outputs from `python scripts/run_pipeline.py` and
`python scripts/run_benchmark.py` on the default 90-day synthetic dataset
(`seed=42` for the single-run, `seed=1000..1024` for benchmarks). They
are committed so the repo can be browsed without first running anything.

| File | What it is |
|---|---|
| `comparison.csv` | Per-anomaly-type Precision / Recall / F1, single seed |
| `alerts_stl_sample.json` | Severity-banded alert log produced by STL |
| `alert_quality.csv` | True-positive rate by severity band, single seed |
| `benchmark_summary.csv` | 25-seed mean ± std (legacy default — service-level) |
| `benchmark_summary_service.csv` | 25-seed mean ± std at the service granularity |
| `benchmark_summary_service_env.csv` | 25-seed mean ± std at (service, env) multi-granularity |

To regenerate:

```bash
python scripts/run_pipeline.py [--granularity service|service_env]
python scripts/run_benchmark.py --seeds 25 [--granularity service|service_env]
```

Outputs land under `outputs/` (which is gitignored).
