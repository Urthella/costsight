"""Project-wide constants for the synthetic dataset and detectors."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SERVICES = [
    ("EC2",        "us-east-1", "BoxUsage",   180.0, 0.18),
    ("S3",         "us-east-1", "Storage",     65.0, 0.10),
    ("RDS",        "eu-west-1", "InstanceHr", 140.0, 0.15),
    ("Lambda",     "us-east-1", "Requests",     5.0, 0.40),
    ("CloudFront", "us-east-1", "DataXfer",    25.0, 0.25),
    ("DynamoDB",   "us-east-1", "ReadUnits",   30.0, 0.20),
    ("EBS",        "eu-west-1", "Storage",     45.0, 0.08),
]

# Environment cost-share weights: prod is the bulk of spend, staging is
# meaningful, dev is a small but noisy slice. Per-service totals match the
# legacy single-env baselines exactly (weights sum to 1.0) so the
# service-level aggregation stays backward-compatible.
ENVIRONMENTS = [
    ("prod",    0.65, 1.0),
    ("staging", 0.25, 1.4),
    ("dev",     0.10, 2.0),
]
# Tuple = (env name, share of service base_cost, noise multiplier).

GRANULARITY_KEYS = ("service", "env")

DEFAULT_DAYS = 90
DEFAULT_SEED = 42

ANOMALY_TYPES = ("point_spike", "level_shift", "gradual_drift")

SEVERITY_BANDS = {
    "LOW":    (0.0, 0.33),
    "MEDIUM": (0.33, 0.66),
    "HIGH":   (0.66, 1.01),
}
