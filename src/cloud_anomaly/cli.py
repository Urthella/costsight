"""Console scripts wired through pyproject.toml's [project.scripts].

After `pip install costsight`, these become available as shell commands:
  costsight-pipeline   — single run, writes outputs/
  costsight-benchmark  — multi-seed Monte Carlo
  costsight-api        — serve the FastAPI app via uvicorn
"""
from __future__ import annotations

import argparse
import sys


def run_pipeline() -> int:
    """`costsight-pipeline [--days 90] [--seed 42] [--scenario default]`."""
    parser = argparse.ArgumentParser(
        prog="costsight-pipeline",
        description="Run the full anomaly-detection pipeline once.",
    )
    parser.add_argument("--days", type=int, default=90, help="Days of synthetic history.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--scenario", type=str, default="default",
        help="Anomaly mix preset (default / drift_heavy / spike_storm / stealth_leak / multi_region / weekend_camouflage / calm).",
    )
    args = parser.parse_args()

    from .pipeline import run as _run
    _run(n_days=args.days, seed=args.seed, scenario=args.scenario)
    return 0


def run_benchmark() -> int:
    """`costsight-benchmark [--seeds 25] [--days 90]`."""
    parser = argparse.ArgumentParser(
        prog="costsight-benchmark",
        description="Run the multi-seed Monte Carlo benchmark.",
    )
    parser.add_argument("--seeds", type=int, default=25, help="Number of random seeds.")
    parser.add_argument("--days", type=int, default=90, help="Days per dataset.")
    args = parser.parse_args()

    from .benchmark import run as _bench
    result = _bench(n_seeds=args.seeds, n_days=args.days)
    print(result.summary.to_string(index=False))
    return 0


def run_api() -> int:
    """`costsight-api [--host 0.0.0.0] [--port 8000]`."""
    parser = argparse.ArgumentParser(
        prog="costsight-api",
        description="Serve the FastAPI surface via uvicorn.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. pip install 'costsight[api]' or 'costsight[dev]'.", file=sys.stderr)
        return 1
    uvicorn.run("cloud_anomaly.api:app", host=args.host, port=args.port, reload=False)
    return 0
