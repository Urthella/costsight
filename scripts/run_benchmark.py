"""CLI: run the multi-seed benchmark and print a Markdown table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.benchmark import render_table, run, write  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    print(f"Running {args.seeds}-seed benchmark on {args.days}-day datasets...")
    result = run(n_seeds=args.seeds, n_days=args.days)
    paths = write(result)
    print(f"\nWrote {paths['summary']}")
    print(f"Wrote {paths['raw']}\n")
    print(render_table(result.summary))
