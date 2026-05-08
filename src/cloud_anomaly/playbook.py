"""Per-anomaly-type runbook hints (what should the FinOps engineer do?).

The pipeline can detect an anomaly, score its severity, and even point
at the dimension that drove it. The next question is "now what?" —
this module answers that question with a short, deterministic recipe
keyed off the detector's output.
"""
from __future__ import annotations

PLAYBOOKS: dict[str, dict[str, str]] = {
    "point_spike": {
        "headline": "Single-day cost explosion — investigate runaway compute or API loop.",
        "checks": (
            "1. Pull CloudTrail for the affected (service, region) on the spike day "
            "and look for unusual API call volume.\n"
            "2. Diff the autoscaling group's desired-capacity vs. actual over the "
            "spike window — was a misconfigured rule triggered?\n"
            "3. Check the SQS / Kinesis backlog at spike onset — a stuck consumer "
            "often produces a billing spike upstream.\n"
            "4. If Lambda: scan the per-function invocation count + duration — a "
            "deployed-this-morning function with a bug usually shows here."
        ),
        "owner": "Service-team on-call",
        "sla": "Acknowledge within 1 hour; mitigate within the same business day.",
    },
    "level_shift": {
        "headline": "Persistent step up — likely a misconfigured deploy or instance class change.",
        "checks": (
            "1. Compare the current instance / task / function size to the "
            "previous-week average (CloudWatch → EC2 → InstanceType).\n"
            "2. Pull the deployment history for the affected service over the "
            "two days preceding the level shift.\n"
            "3. Verify the autoscaling group's max-size hasn't been raised "
            "(common after a load test that wasn't reverted).\n"
            "4. If RDS: check parameter-group changes (e.g. enabling Performance "
            "Insights with full retention silently doubles writes)."
        ),
        "owner": "Service-team lead + FinOps",
        "sla": "Mitigate within 48 hours — every day of inaction compounds.",
    },
    "gradual_drift": {
        "headline": "Slow upward creep — usually unmanaged data accumulation or zombie resources.",
        "checks": (
            "1. S3 / EBS: look for log buckets without lifecycle policies — "
            "GetBucketLifecycleConfiguration should not return NoSuchLifecycleConfiguration.\n"
            "2. CloudWatch Logs: any log group with retention = 'Never expire'? "
            "Set it to 30 days unless there's a compliance requirement.\n"
            "3. Snapshots: scan EC2 / RDS snapshots older than 90 days; tag "
            "owners and confirm before deletion.\n"
            "4. Forgotten dev environments: cross-reference cost per "
            "tag_environment — staging or dev growing faster than prod is a "
            "smell (especially for unattended PR-environment factories)."
        ),
        "owner": "FinOps weekly review",
        "sla": "Open a tracking ticket the same day; resolve in the next sprint.",
    },
    "multi_detector_consensus": {
        "headline": "Multiple detectors flag the same day — high confidence anomaly.",
        "checks": (
            "Treat this as the highest-priority class regardless of severity "
            "score. The fact that statistical (Z-Score), decomposition (STL), "
            "and density (Isolation Forest) approaches all agree means the "
            "signal is unlikely to be noise.\n"
            "Run the per-anomaly-type playbook for the matched type, but "
            "page the on-call lead immediately rather than waiting for the "
            "scheduled triage window."
        ),
        "owner": "On-call lead",
        "sla": "Acknowledge within 30 minutes.",
    },
}


def get(anomaly_type: str) -> dict[str, str]:
    """Return the playbook for a given anomaly_type. Falls back gracefully."""
    return PLAYBOOKS.get(
        anomaly_type,
        {
            "headline": f"Unknown anomaly type: {anomaly_type}.",
            "checks": "No playbook entry yet — escalate to FinOps lead for triage.",
            "owner": "FinOps",
            "sla": "Same business day.",
        },
    )
