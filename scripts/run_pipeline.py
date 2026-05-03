"""CLI entry point for running the full pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.pipeline import run  # noqa: E402

if __name__ == "__main__":
    artifacts = run()
    print("\n=== Detector comparison (P/R by anomaly type) ===")
    print(artifacts["comparison"].to_string(index=False))
    print(f"\nArtifacts written to {ROOT / 'outputs'}")
