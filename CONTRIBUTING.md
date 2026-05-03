# Contributing to costsight

Thanks for opening this file. costsight is a course project, but the
codebase is small and welcoming to contributions — small fixes, new
detectors, dashboard polish, and additional anomaly types are all in
scope.

## Local setup

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python scripts/run_pipeline.py
```

If `pytest` is green and `run_pipeline.py` finishes without raising,
your environment is good.

## Adding a new detector

1. Create `src/cloud_anomaly/detectors/<name>.py` exposing
   `detect(long_df) -> pd.DataFrame` with the standard schema:
   `date, service, cost, score, is_anomaly`.
2. Register it in `src/cloud_anomaly/detectors/__init__.py` by adding
   to the `DETECTORS` mapping.
3. Add a smoke test path in `tests/test_pipeline.py` if the detector
   has unusual edge cases.
4. Re-run `python scripts/run_benchmark.py --seeds 25` to refresh the
   numbers and update `examples/benchmark_summary.csv`.
5. Open a PR — CI will run pytest + the full pipeline on Python 3.11
   and 3.12 automatically.

## Adding a new anomaly type

1. Extend `InjectedAnomaly` semantics in
   `src/cloud_anomaly/synthetic_data.py` if your type cannot be
   expressed as a multiplicative pattern over a window.
2. Append to `_default_anomalies()` so the type appears in the
   committed dataset.
3. The evaluator (`evaluate_by_type`) picks up new types automatically
   from the ground-truth `anomaly_type` column.

## Code style

- Type hints on public functions; `from __future__ import annotations`
  at the top of every module.
- No comments narrating *what* the code does; only *why* (a non-obvious
  invariant or a workaround).
- Small, single-responsibility functions over generic abstractions.
- Tests live in `tests/` and run via `pytest -q`.

## Reporting issues

GitHub Issues at <https://github.com/Urthella/costsight/issues>. For
detector-quality discussions, please attach the seed and parameter
combination that reproduces the behaviour.
