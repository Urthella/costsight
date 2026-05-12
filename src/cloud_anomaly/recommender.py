"""Cost-optimization recommender — turns observations into actionable savings.

Where the anomaly detector answers *"what happened?"*, this module
answers *"what should we do about it?"* — independently of any active
anomaly. It scans the CUR for the four most common FinOps wins:

  1. Reserved Instance / Savings Plan candidates (sustained compute).
  2. Idle storage / over-provisioned EBS volumes.
  3. Untagged spend → fixed with tag governance.
  4. Cross-region traffic that could be served closer to the user.

Each finding includes an estimated monthly savings and a concrete
remediation step so a FinOps engineer can prioritize at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pricing import lookup


@dataclass
class Recommendation:
    """One actionable finding the dashboard surfaces."""
    category: str
    service: str
    region: str
    impact_usd_per_month: float
    confidence: str          # "high" / "medium" / "low"
    action: str
    rationale: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "category": self.category,
            "service": self.service,
            "region": self.region,
            "impact_usd_per_month": round(self.impact_usd_per_month, 2),
            "confidence": self.confidence,
            "action": self.action,
            "rationale": self.rationale,
        }


def reserved_instance_candidates(cur_df: pd.DataFrame) -> list[Recommendation]:
    """Find services with low-volatility, sustained spend → RI / SP candidates.

    Heuristic: a service whose daily cost has std/mean < 0.20 over the
    window AND whose daily spend exceeds $50 is a textbook RI candidate
    (reserved-instance pricing typically saves 30-40% vs on-demand).
    """
    if cur_df.empty:
        return []
    daily = (
        cur_df.assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby(["date", "service", "region"], as_index=False)["cost"].sum()
    )
    findings: list[Recommendation] = []
    for (svc, region), group in daily.groupby(["service", "region"]):
        if svc not in {"EC2", "RDS", "ElastiCache", "Redshift"}:
            continue  # only services with RI / SP coverage
        mean = float(group["cost"].mean())
        std = float(group["cost"].std() or 0.0)
        if mean < 50 or std / max(mean, 1e-6) > 0.20:
            continue
        monthly_on_demand = mean * 30
        savings = monthly_on_demand * 0.35  # typical 3-year all-upfront discount
        findings.append(Recommendation(
            category="Reserved capacity",
            service=svc, region=region,
            impact_usd_per_month=savings,
            confidence="high" if std / mean < 0.10 else "medium",
            action=f"Purchase a 1-yr no-upfront Savings Plan for {svc} in {region} "
                   f"covering ~${mean*24:.0f}/day baseline.",
            rationale=f"Daily spend std/mean = {std/mean:.0%} over the window — "
                      f"low-volatility, sustained workload typical of RI/SP candidates.",
        ))
    return sorted(findings, key=lambda r: -r.impact_usd_per_month)


def idle_storage_candidates(cur_df: pd.DataFrame) -> list[Recommendation]:
    """Detect EBS / Glacier-eligible S3 patterns.

    Heuristic: EBS spend in a region with no proportional EC2 / RDS
    spend suggests detached volumes (paid for, not attached). S3 spend
    that has been *exactly flat* for >30 days is a candidate for
    lifecycle policy → IA / Glacier transition.
    """
    if cur_df.empty:
        return []
    daily = (
        cur_df.assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby(["date", "service", "region"], as_index=False)["cost"].sum()
    )
    findings: list[Recommendation] = []

    # EBS-without-compute heuristic.
    ebs = daily[daily["service"] == "EBS"].groupby("region", as_index=False)["cost"].sum()
    compute_regions = set(
        daily[daily["service"].isin(["EC2", "RDS"])]["region"].unique()
    )
    for _, row in ebs.iterrows():
        if row["region"] not in compute_regions and row["cost"] > 0:
            monthly = row["cost"] / max(daily["date"].nunique(), 1) * 30
            findings.append(Recommendation(
                category="Idle storage",
                service="EBS", region=row["region"],
                impact_usd_per_month=monthly * 0.85,  # most can be deleted
                confidence="medium",
                action=f"Audit EBS volumes in {row['region']}: enumerate detached "
                       f"volumes (DescribeVolumes filter=Attachment.Status:detached) "
                       f"and delete after a 7-day tag-confirm window.",
                rationale=f"EBS spend in {row['region']} without matching EC2/RDS "
                          f"spend — strong signal that volumes are detached.",
            ))

    # S3 spend that hasn't changed across the window → tier candidate.
    s3 = daily[daily["service"] == "S3"]
    for (region,), group in s3.groupby(["region"]):
        if len(group) < 30:
            continue
        flatness = (group["cost"].std() / max(group["cost"].mean(), 1e-6)) if group["cost"].mean() else 1.0
        if flatness < 0.03 and group["cost"].mean() > 20:
            monthly = group["cost"].mean() * 30
            findings.append(Recommendation(
                category="Storage tiering",
                service="S3", region=region,
                impact_usd_per_month=monthly * 0.45,  # IA discount vs Standard
                confidence="medium",
                action=f"Add an S3 lifecycle policy in {region}: transition objects "
                       f"older than 30 days to S3 Infrequent Access (~46% cheaper).",
                rationale=f"S3 spend in {region} has been flat (std/mean = {flatness:.1%}) "
                          f"over the window — objects are likely cold-stored at Standard tier.",
            ))
    return sorted(findings, key=lambda r: -r.impact_usd_per_month)


def untagged_spend(cur_df: pd.DataFrame) -> list[Recommendation]:
    """Quantify the cost of unowned / untagged resources."""
    if cur_df.empty or "tag_team" not in cur_df.columns:
        return []
    df = cur_df.assign(date=lambda d: pd.to_datetime(d["date"]))
    untagged = df[(df["tag_team"] == "") | (df["tag_team"].isna())]
    if untagged.empty:
        return []
    monthly = untagged["cost"].sum() / max(df["date"].nunique(), 1) * 30
    if monthly < 1:
        return []
    return [Recommendation(
        category="Tag governance",
        service="(multiple)", region="(multiple)",
        impact_usd_per_month=monthly,
        confidence="high",
        action="Enforce a deploy-time tagging policy (AWS Organizations SCP or "
               "Config Rule); back-fill historical resources via Resource Groups Tagging API.",
        rationale=f"${untagged['cost'].sum():,.0f} of spend over the window has no "
                  f"tag_team — that money has no owner and can't be charged-back.",
    )]


def cross_region_traffic(cur_df: pd.DataFrame) -> list[Recommendation]:
    """Detect heavy DataTransfer spend that might benefit from CloudFront / VPC peering."""
    if cur_df.empty:
        return []
    df = cur_df.assign(date=lambda d: pd.to_datetime(d["date"]))
    transfer = df[df["service"].isin(["DataTransfer", "CloudFront"])].groupby(
        ["region"], as_index=False
    )["cost"].sum()
    findings: list[Recommendation] = []
    for _, row in transfer.iterrows():
        monthly = row["cost"] / max(df["date"].nunique(), 1) * 30
        if monthly < 30:
            continue
        findings.append(Recommendation(
            category="Network egress",
            service="DataTransfer", region=row["region"],
            impact_usd_per_month=monthly * 0.35,
            confidence="low",
            action=f"Audit {row['region']} egress: replace cross-region S3 transfers "
                   f"with VPC peering, and front public traffic with CloudFront "
                   f"(cheaper per-GB beyond 10 TB).",
            rationale=f"DataTransfer + CloudFront spend in {row['region']} is "
                      f"${monthly:.0f}/month, a tier where the egress optimizations "
                      f"materially pay for themselves.",
        ))
    return findings


def all_recommendations(cur_df: pd.DataFrame) -> pd.DataFrame:
    """Run every heuristic and return a single sorted DataFrame."""
    findings: list[Recommendation] = (
        reserved_instance_candidates(cur_df)
        + idle_storage_candidates(cur_df)
        + untagged_spend(cur_df)
        + cross_region_traffic(cur_df)
    )
    if not findings:
        return pd.DataFrame(columns=[
            "category", "service", "region", "impact_usd_per_month",
            "confidence", "action", "rationale",
        ])
    return pd.DataFrame([r.to_dict() for r in findings]).sort_values(
        "impact_usd_per_month", ascending=False
    ).reset_index(drop=True)
