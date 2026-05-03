"""CLI entry point for running the full pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.pipeline import run  # noqa: E402

GRANULARITIES = {
    "service": ("service",),
    "service_env": ("service", "env"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularity",
        choices=sorted(GRANULARITIES),
        default="service",
        help="service = legacy per-service runs, service_env = per (service, env) multi-granularity",
    )
    args = parser.parse_args()

    artifacts = run(group_keys=GRANULARITIES[args.granularity])
    print(f"\n=== Detector comparison (P/R by anomaly type) — {args.granularity} ===")
    print(artifacts["comparison"].to_string(index=False))
    if not artifacts["alert_quality"].empty:
        print("\n=== Alert quality by severity band ===")
        print(artifacts["alert_quality"].to_string(index=False))
    print(f"\nArtifacts written to {ROOT / 'outputs'}")
