"""Webhook + SNS-shaped alert notification helpers (Slack / email / PagerDuty).

Production deploy would route alerts through SNS → subscribed
endpoints. Keeping the *payload shape* identical to that flow means the
dashboard's "send sample alert" button doubles as documentation for
the production integration.

No live HTTP is made here unless an explicit webhook URL is supplied —
the dashboard exposes that field; tests stub it out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AlertPayload:
    """The canonical shape we'd push to SNS / Slack / PagerDuty."""
    title: str
    severity: str
    service: str
    date: str
    cost: float
    summary: str
    detector: str
    runbook: str

    def to_slack_block(self) -> dict[str, Any]:
        """Slack Block Kit message — survives copy-paste into Slack's UI."""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"🚨 {self.title}"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Service:*\n{self.service}"},
                        {"type": "mrkdwn", "text": f"*Date:*\n{self.date}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{self.severity}"},
                        {"type": "mrkdwn", "text": f"*Cost:*\n${self.cost:,.2f}"},
                        {"type": "mrkdwn", "text": f"*Detector:*\n{self.detector}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Summary:* {self.summary}"},
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"_Runbook:_ {self.runbook}"}],
                },
            ]
        }

    def to_email_text(self) -> str:
        return (
            f"Subject: [{self.severity}] {self.title}\n\n"
            f"Service: {self.service}\n"
            f"Date: {self.date}\n"
            f"Cost: ${self.cost:,.2f}\n"
            f"Detector: {self.detector}\n\n"
            f"Summary:\n  {self.summary}\n\n"
            f"Runbook:\n  {self.runbook}\n"
        )

    def to_sns(self) -> dict[str, str]:
        """SNS Publish payload — maps cleanly to ``boto3.client('sns').publish(**…)``."""
        return {
            "Subject": f"[{self.severity}] {self.title}"[:99],
            "Message": json.dumps({
                "title": self.title,
                "severity": self.severity,
                "service": self.service,
                "date": self.date,
                "cost": self.cost,
                "summary": self.summary,
                "detector": self.detector,
                "runbook": self.runbook,
            }, indent=2),
        }


def build_payload_from_alert(
    alert_row: dict[str, Any] | pd.Series,
    *,
    detector: str = "stl",
    runbook: str = "",
) -> AlertPayload:
    """Adapt an alerts.build_alerts row → AlertPayload."""
    row = dict(alert_row) if not isinstance(alert_row, dict) else alert_row
    date_value = row.get("date", "")
    if isinstance(date_value, pd.Timestamp):
        date_str = date_value.strftime("%Y-%m-%d")
    else:
        date_str = str(date_value)
    severity = str(row.get("severity", "LOW")).upper()
    return AlertPayload(
        title=f"Cost anomaly on {row.get('service','?')} ({severity})",
        severity=severity,
        service=str(row.get("service", "?")),
        date=date_str,
        cost=float(row.get("cost", 0.0)),
        summary=str(row.get("summary", row.get("flagged_by", ""))),
        detector=str(detector),
        runbook=str(runbook or "see REPORT § 4.1 for the production runbook"),
    )


def send_webhook(payload: AlertPayload, url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """POST a Slack-shaped payload to ``url``; returns a small status dict.

    Uses ``httpx`` (already a dependency for FastAPI tests) so there's
    no extra runtime weight. Network failures are swallowed and reported
    in the return dict — the caller can render the result without
    crashing the dashboard.
    """
    try:
        import httpx
    except ImportError:
        return {"status": "skipped", "error": "httpx not installed"}

    try:
        response = httpx.post(url, json=payload.to_slack_block(), timeout=timeout)
        return {
            "status": "ok" if response.status_code < 400 else "error",
            "code": response.status_code,
            "preview": response.text[:200],
        }
    except Exception as exc:  # network or DNS failure
        return {"status": "error", "error": str(exc)}
