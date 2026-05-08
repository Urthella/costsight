"""Real-AWS-CUR ingestion adapter.

AWS Cost & Usage Report (CUR) files use a verbose, slash-separated
column naming convention (e.g. ``lineItem/UnblendedCost``). This module
maps that schema onto the project's internal long-format
``(date, service, region, usage_type, cost, [tag_team, tag_environment])``
without introducing any AWS-SDK runtime dependency — we just read the
CSV / Parquet that AWS drops in your S3 bucket.

Tested against:
  - Resource-level CUR ("Hourly + Resources + Tags")
  - Daily aggregated CUR
  - The redacted public sample committed to ``examples/aws_cur_sample.csv``

Usage::

    from cloud_anomaly.cur_loader import load_cur_csv
    long_df = load_cur_csv("path/to/aws_cur.csv")
    detections = stl_detect(long_df)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# AWS CUR product codes → the short names used elsewhere in the project.
# Add new mappings here when onboarding more services.
PRODUCT_CODE_MAP: dict[str, str] = {
    "AmazonEC2": "EC2",
    "AmazonS3": "S3",
    "AmazonRDS": "RDS",
    "AWSLambda": "Lambda",
    "AmazonCloudFront": "CloudFront",
    "AmazonDynamoDB": "DynamoDB",
    "AmazonEBS": "EBS",
    "AmazonECS": "ECS",
    "AmazonEKS": "EKS",
    "AWSDataTransfer": "DataTransfer",
    "AmazonRoute53": "Route53",
    "AmazonSNS": "SNS",
    "AmazonSQS": "SQS",
    "AWSGlue": "Glue",
    "AmazonAthena": "Athena",
    "AmazonECR": "ECR",
    "AmazonElasticCache": "ElastiCache",
    "AmazonKinesis": "Kinesis",
    "AmazonSageMaker": "SageMaker",
    "AmazonRedshift": "Redshift",
}

# CUR columns we consume. Each key is one of our internal columns; the
# value is a list of CUR column names to try (first match wins).
COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "date": (
        "lineItem/UsageStartDate",
        "line_item_usage_start_date",
        "lineitem/usagestartdate",
    ),
    "product_code": (
        "lineItem/ProductCode",
        "line_item_product_code",
        "lineitem/productcode",
    ),
    "region": (
        "product/region",
        "product_region",
        "product/regionCode",
    ),
    "usage_type": (
        "lineItem/UsageType",
        "line_item_usage_type",
        "lineitem/usagetype",
    ),
    "cost": (
        "lineItem/UnblendedCost",
        "line_item_unblended_cost",
        "lineitem/unblendedcost",
    ),
    "tag_team": (
        "resourceTags/user:Team",
        "resource_tags_user_team",
        "resourceTags/user:team",
    ),
    "tag_environment": (
        "resourceTags/user:Environment",
        "resource_tags_user_environment",
        "resourceTags/user:environment",
    ),
}


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_cur_csv(path: str | Path) -> pd.DataFrame:
    """Load an AWS CUR CSV / Parquet file as a long-format DataFrame.

    Returns columns:
      date, service, region, usage_type, cost, tag_team, tag_environment.
    Tag columns may be filled with the empty string when the source CUR
    has no resource tags. Costs are aggregated to one row per
    (date, service, region, usage_type) by summing UnblendedCost.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    cols = {key: _pick_column(raw, cands) for key, cands in COLUMN_CANDIDATES.items()}
    missing = [k for k in ("date", "product_code", "cost") if cols[k] is None]
    if missing:
        raise ValueError(
            f"AWS CUR file at {path} is missing required columns: {missing}. "
            f"Available columns: {sorted(raw.columns)[:20]}…"
        )

    # Normalize and coerce types.
    df = pd.DataFrame()
    parsed_dates = pd.to_datetime(raw[cols["date"]], errors="coerce", utc=True)
    df["date"] = parsed_dates.dt.tz_localize(None).dt.normalize()
    df["service"] = raw[cols["product_code"]].map(
        lambda v: PRODUCT_CODE_MAP.get(str(v), str(v))
    )
    df["region"] = raw[cols["region"]].fillna("global") if cols["region"] else "global"
    df["usage_type"] = raw[cols["usage_type"]].fillna("Other") if cols["usage_type"] else "Other"
    df["cost"] = pd.to_numeric(raw[cols["cost"]], errors="coerce").fillna(0.0)

    df["tag_team"] = (
        raw[cols["tag_team"]].fillna("").astype(str)
        if cols["tag_team"] else ""
    )
    df["tag_environment"] = (
        raw[cols["tag_environment"]].fillna("").astype(str)
        if cols["tag_environment"] else ""
    )

    df = df.dropna(subset=["date"])
    aggregated = (
        df.groupby(
            ["date", "service", "region", "usage_type", "tag_team", "tag_environment"],
            as_index=False,
        )["cost"]
        .sum()
    )
    return aggregated.sort_values(["date", "service", "region", "usage_type"]).reset_index(
        drop=True
    )


def write_internal_cur(long_df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Persist a CUR-loader-output long_df to the project's RAW_DIR layout.

    Writes ``cur_synthetic.{csv,parquet}`` so the rest of the pipeline
    can consume it untouched.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "cur_synthetic.csv", index=False)
    long_df.to_parquet(out_dir / "cur_synthetic.parquet", index=False)
    return out_dir
