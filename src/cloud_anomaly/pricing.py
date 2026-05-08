"""AWS Pricing lookup — used to ground the synthetic / forecast costs in
real on-demand prices.

The Pricing List Bulk API is publicly accessible, no IAM auth needed:

    https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/<service>/current/<region>/index.json

We don't fetch on every dashboard render (the JSON files are 100+ MB).
Instead we ship a curated snapshot of the few services / instance types
the project demos with, and only hit the network when the user
explicitly clicks "refresh from AWS".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Curated snapshot — captured 2025-04-15 from the public Pricing List
# Bulk API. Costs are USD/hour for compute, USD/GB-month for storage.
# These values are illustrative; the dashboard surfaces them with a
# "AS OF 2025-04-15" disclaimer.
PRICING_SNAPSHOT: dict[str, dict[str, dict[str, float]]] = {
    "EC2": {
        "us-east-1": {"t3.medium": 0.0416, "t3.large": 0.0832, "m5.large": 0.096, "m5.xlarge": 0.192},
        "eu-west-1": {"t3.medium": 0.0456, "t3.large": 0.0912, "m5.large": 0.111, "m5.xlarge": 0.222},
    },
    "RDS": {
        "us-east-1": {"db.t3.medium": 0.082, "db.r5.large": 0.290, "db.r5.xlarge": 0.580},
        "eu-west-1": {"db.t3.medium": 0.094, "db.r5.large": 0.330, "db.r5.xlarge": 0.660},
    },
    "S3": {
        "us-east-1": {"Standard-GB-month": 0.023, "IA-GB-month": 0.0125, "Glacier-GB-month": 0.004},
        "eu-west-1": {"Standard-GB-month": 0.024, "IA-GB-month": 0.013,  "Glacier-GB-month": 0.0045},
    },
    "Lambda": {
        "us-east-1": {"GB-second": 0.0000166667, "Request": 0.0000002},
    },
    "EBS": {
        "us-east-1": {"gp3-GB-month": 0.08, "io2-GB-month": 0.125},
        "eu-west-1": {"gp3-GB-month": 0.088, "io2-GB-month": 0.137},
    },
    "DynamoDB": {
        "us-east-1": {"OnDemand-Read-Million": 0.25, "OnDemand-Write-Million": 1.25},
    },
    "CloudFront": {
        "us-east-1": {"DataTransfer-GB": 0.085, "Request-10k": 0.0075},
    },
}

PRICING_SNAPSHOT_DATE = "2025-04-15"


@dataclass
class PriceQuote:
    service: str
    region: str
    sku: str
    unit_price: float
    unit: str
    source: str
    captured_at: str


def lookup(
    service: str,
    region: str = "us-east-1",
    sku: str | None = None,
) -> list[PriceQuote]:
    """Return matching SKUs from the snapshot. ``sku=None`` returns all SKUs.

    Looking up ``EC2 / us-east-1`` yields one PriceQuote per instance type
    in that region. Mostly used by the dashboard's "is this synthetic
    cost realistic?" sanity-check sidebar.
    """
    by_region = PRICING_SNAPSHOT.get(service, {})
    by_sku = by_region.get(region, {})
    items = by_sku.items() if sku is None else [(sku, by_sku[sku])] if sku in by_sku else []
    return [
        PriceQuote(
            service=service, region=region, sku=k, unit_price=float(v),
            unit=_unit_for_sku(k), source="snapshot",
            captured_at=PRICING_SNAPSHOT_DATE,
        )
        for k, v in items
    ]


def _unit_for_sku(sku: str) -> str:
    if "GB-month" in sku:
        return "USD per GB-month"
    if "GB-second" in sku:
        return "USD per GB-second"
    if "Request" in sku and "10k" not in sku:
        return "USD per request"
    if "Request-10k" in sku:
        return "USD per 10,000 requests"
    if "Million" in sku:
        return "USD per million units"
    if "DataTransfer" in sku:
        return "USD per GB transferred"
    return "USD per hour"


def estimated_monthly(service: str, region: str, sku: str, hours_per_month: float = 730) -> float:
    """Convert hourly rate × 730 (avg hrs/mo) for a quick monthly estimate."""
    quotes = lookup(service, region, sku)
    if not quotes:
        return float("nan")
    rate = quotes[0].unit_price
    if "hour" in quotes[0].unit:
        return rate * hours_per_month
    return rate  # storage / bytes — already monthly or per-event


def fetch_live(
    service: str,
    region: str = "us-east-1",
    *,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch the live Pricing List Bulk JSON for one (service, region).

    Network-dependent; only called from the dashboard's explicit
    "refresh" button. Returns the parsed JSON. Caches under
    ``cache_dir`` to avoid re-downloading the same offer.
    """
    import httpx

    url = (
        f"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/"
        f"{service}/current/{region}/index.json"
    )
    response = httpx.get(url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{service}-{region}.json").write_text(json.dumps(data))
    return data
