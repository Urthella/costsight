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

# (service) → (tag_team, tag_environment).
# Mirrors a typical org where backend services live in prod and analytics
# services live in staging - gives the attribution layer something to
# pivot on beyond region / usage_type.
SERVICE_TAGS: dict[str, tuple[str, str]] = {
    "EC2":        ("backend",   "prod"),
    "S3":         ("data",      "prod"),
    "RDS":        ("backend",   "prod"),
    "Lambda":     ("platform",  "staging"),
    "CloudFront": ("frontend",  "prod"),
    "DynamoDB":   ("backend",   "staging"),
    "EBS":        ("platform",  "prod"),
}

DEFAULT_DAYS = 90
DEFAULT_SEED = 42

ANOMALY_TYPES = ("point_spike", "level_shift", "gradual_drift")

# Calibrated to the range the severity formula actually produces. The
# (0.4 + 0.6*x) factors plus a sustained anomaly inflating its own mean cap
# achievable severity near ~0.5, so the old 0.66 HIGH cut-off was unreachable -
# nothing ever scored HIGH. These cut-points put the worst anomalies in HIGH
# while keeping HIGH selective (a few per scenario).
SEVERITY_BANDS = {
    "LOW":    (0.0, 0.20),
    "MEDIUM": (0.20, 0.40),
    "HIGH":   (0.40, 1.01),
}
