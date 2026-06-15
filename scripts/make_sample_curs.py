"""Generate realistic AWS CUR-format sample CSVs from the synthetic generator.

These give the dashboard's "Upload AWS CUR (.csv)" button demo-ready test
files of meaningful size across several anomaly scenarios - handy when a
reviewer asks to try the tool on different data.

Run:
    python scripts/make_sample_curs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from cloud_anomaly.synthetic_data import generate  # noqa: E402
from cloud_anomaly.cur_loader import PRODUCT_CODE_MAP  # noqa: E402

REVERSE = {short: code for code, short in PRODUCT_CODE_MAP.items()}
OUT = ROOT / "examples"

CONFIGS = [
    # The headline "sample" a reviewer is most likely to grab: 90 days of
    # spike-storm data so the upload path immediately fills every view. Short
    # files (the old 4-day fixture lives in cur_loader_fixture.csv) flag nothing
    # because the detectors need ~2 weeks of history.
    ("aws_cur_sample", dict(n_days=90, seed=7, scenario="spike_storm")),
    # The "see everything" upload: sustained level shifts (HIGH severity),
    # spikes and drift across services so every view fills with strong signal.
    ("cur_showcase_90d", dict(n_days=90, seed=9, scenario="showcase")),
    ("cur_default_90d", dict(n_days=90, seed=42, scenario="default")),
    ("cur_spike_storm_60d", dict(n_days=60, seed=7, scenario="spike_storm")),
    ("cur_stealth_leak_90d", dict(n_days=90, seed=3, scenario="stealth_leak")),
    ("cur_multi_region_90d", dict(n_days=90, seed=5, scenario="multi_region")),
    ("cur_calm_60d", dict(n_days=60, seed=11, scenario="calm")),
]


def to_cur(cur: pd.DataFrame) -> pd.DataFrame:
    """Map the internal long format onto AWS CUR column names."""
    df = pd.DataFrame()
    df["lineItem/UsageStartDate"] = pd.to_datetime(cur["date"]).dt.strftime(
        "%Y-%m-%dT00:00:00Z"
    )
    df["lineItem/ProductCode"] = cur["service"].map(lambda s: REVERSE.get(s, s))
    df["product/region"] = cur["region"]
    df["lineItem/UsageType"] = cur["usage_type"]
    df["lineItem/UnblendedCost"] = cur["cost"].round(4)
    df["resourceTags/user:Team"] = cur["tag_team"]
    df["resourceTags/user:Environment"] = cur["tag_environment"]
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, kw in CONFIGS:
        cur, _, _ = generate(**kw)
        path = OUT / f"{name}.csv"
        to_cur(cur).to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)} ({len(cur):,} rows, {kw['scenario']})")


if __name__ == "__main__":
    main()
