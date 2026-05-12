"""Tag governance — quantify and prioritize tag debt.

The single biggest unforced FinOps error is **untagged spend** — money
that can't be attributed to a team or environment, so nobody owns it,
so nobody optimizes it. This module computes:

  * Per-tag coverage rate (what % of spend has each required tag).
  * Tag debt in USD ($ of untagged spend over the window).
  * Per-service worst offenders ranked by dollar impact.
  * Policy-as-code stub showing what enforcement would look like.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_TAGS = ("tag_team", "tag_environment")


@dataclass
class TagReport:
    coverage: pd.DataFrame
    debt_usd: float
    worst_services: pd.DataFrame
    policy_yaml: str


def evaluate_tagging(cur_df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_TAGS) -> TagReport:
    """Compute coverage + dollar debt for each required tag.

    A row is "untagged" w.r.t. a tag if the value is the empty string,
    NaN, or the literal "unknown". Coverage is the cost-weighted
    percentage of rows with non-empty values.
    """
    if cur_df.empty:
        return TagReport(
            coverage=pd.DataFrame(columns=["tag", "covered_pct", "untagged_usd"]),
            debt_usd=0.0,
            worst_services=pd.DataFrame(columns=["service", "untagged_usd", "share_of_service_pct"]),
            policy_yaml="",
        )

    df = cur_df.copy()
    total_cost = float(df["cost"].sum())

    rows = []
    for tag in required:
        if tag not in df.columns:
            rows.append({"tag": tag, "covered_pct": 0.0, "untagged_usd": total_cost})
            continue
        is_untagged = df[tag].isna() | (df[tag] == "") | (df[tag].astype(str).str.lower() == "unknown")
        untagged_cost = float(df.loc[is_untagged, "cost"].sum())
        covered = (1 - untagged_cost / total_cost) * 100 if total_cost else 0.0
        rows.append({
            "tag": tag,
            "covered_pct": round(covered, 1),
            "untagged_usd": round(untagged_cost, 2),
        })
    coverage = pd.DataFrame(rows)

    # Worst offenders: services with the largest untagged $ on ANY required tag.
    worst_rows = []
    for service, group in df.groupby("service"):
        svc_total = float(group["cost"].sum())
        untagged = 0.0
        for tag in required:
            if tag not in group.columns:
                untagged = svc_total
                break
            is_u = group[tag].isna() | (group[tag] == "") | (group[tag].astype(str).str.lower() == "unknown")
            untagged = max(untagged, float(group.loc[is_u, "cost"].sum()))
        if untagged > 0:
            worst_rows.append({
                "service": service,
                "untagged_usd": round(untagged, 2),
                "share_of_service_pct": round(untagged / svc_total * 100, 1) if svc_total else 0.0,
            })
    if worst_rows:
        worst = (
            pd.DataFrame(worst_rows)
            .sort_values("untagged_usd", ascending=False)
            .reset_index(drop=True)
        )
    else:
        worst = pd.DataFrame(
            columns=["service", "untagged_usd", "share_of_service_pct"]
        )

    debt = float(coverage["untagged_usd"].max())

    policy = _emit_policy_yaml(required)

    return TagReport(coverage=coverage, debt_usd=debt, worst_services=worst, policy_yaml=policy)


def _emit_policy_yaml(required: tuple[str, ...]) -> str:
    """A copy-pasteable AWS Config / SCP rule that enforces these tags."""
    tag_list = "\n".join(f"      - {t.removeprefix('tag_').capitalize()}" for t in required)
    return f"""# AWS Config / SCP policy stub — enforce the required tags at deploy-time.
# Save as required-tags.yaml and apply via Config Rules or SCP.
Type: AWS::Config::ConfigRule
Properties:
  ConfigRuleName: required-tags
  Source:
    Owner: AWS
    SourceIdentifier: REQUIRED_TAGS
  InputParameters:
    tag1Key: {required[0].removeprefix('tag_').capitalize()}
    tag2Key: {required[1].removeprefix('tag_').capitalize() if len(required) > 1 else ''}
  Scope:
    ComplianceResourceTypes:
      - AWS::EC2::Instance
      - AWS::S3::Bucket
      - AWS::RDS::DBInstance
      - AWS::EBS::Volume
      - AWS::Lambda::Function
"""
