"""LLM-powered natural-language root-cause explanation for alerts.

Given an alert + its attribution + the last 7 days of CUR rows, ask
Claude (or any Anthropic-compatible API) to write a 3-4 sentence
hypothesis about the cause. The pipeline already has the *deterministic*
hint ("us-east-1 region drove 100% of the increase"); this layer is the
**narrative** that a FinOps engineer would paste into a ticket.

API key handling:
  * Reads ANTHROPIC_API_KEY from the environment.
  * If absent, the module still works — it returns a *templated*
    explanation produced from the same input, so the dashboard never
    breaks on a missing key. The template version is clearly labeled
    so reviewers can tell which mode produced the text.

Cost control:
  * Default model is claude-haiku-4-5 (cheapest fast model). Caller can
    override via the `model` argument.
  * Output is capped at max_tokens=400 (≈ 300 words).
  * Built-in result cache keyed by (alert_date, service, severity) so
    the dashboard doesn't re-burn an API call on every rerender.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd


@dataclass
class Explanation:
    text: str
    source: str           # "anthropic" | "template" | "cache"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


# Module-level cache so the dashboard's reruns don't re-spend API tokens.
_CACHE: dict[str, Explanation] = {}


def _cache_key(alert: dict, attribution: dict) -> str:
    payload = (
        f"{alert.get('date','')}|{alert.get('service','')}|"
        f"{alert.get('severity','')}|{attribution.get('top_value','')}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _template_explanation(
    alert: dict,
    attribution: dict,
    cur_window: pd.DataFrame,
) -> str:
    """Deterministic fallback when no API key is configured.

    Produces a paragraph that reads like the LLM version so the
    dashboard layout doesn't shift between modes.
    """
    service = alert.get("service", "?")
    date = alert.get("date", "?")
    severity = alert.get("severity", "?")
    cost = float(alert.get("cost", 0.0))
    summary = str(attribution.get("summary", ""))
    top_dim = str(attribution.get("top_dimension", "")).replace("_", " ")
    top_val = str(attribution.get("top_value", ""))
    delta_pct = 0.0
    base = float(attribution.get("baseline_cost", 0.0))
    if base > 0:
        delta_pct = (cost - base) / base * 100

    cur_summary = ""
    if not cur_window.empty:
        n_days = cur_window["date"].nunique()
        cur_summary = (
            f" The trailing {n_days}-day window for {service} shows "
            f"costs of "
            f"${cur_window['cost'].mean():,.0f}/day on average "
            f"(σ ≈ ${cur_window['cost'].std():,.0f}); the anomaly day "
            f"breaks ~{abs(delta_pct):.0f}% above that mean."
        )

    return (
        f"On **{date}**, {service} spend reached **${cost:,.2f}** "
        f"(severity={severity}). {summary} The dominant contributor was "
        f"`{top_dim}={top_val}`.{cur_summary}\n\n"
        f"Most likely cause: a sudden change in load or configuration "
        f"affecting that {top_dim} — for example, an autoscaling rule "
        f"raising capacity, a deploy doubling resource footprint, or a "
        f"misbehaving consumer driving abnormal API call volume. Cross-"
        f"reference CloudTrail entries for `{service}` in the 24 hours "
        f"preceding {date} and the deploy history of services owning "
        f"that {top_dim}."
    )


def _call_anthropic(
    alert: dict,
    attribution: dict,
    cur_window: pd.DataFrame,
    *,
    model: str,
    api_key: str,
) -> Explanation | None:
    """Make a live call to the Anthropic API. Returns None on any failure."""
    try:
        import anthropic
    except ImportError:
        return None

    prompt = (
        "You are a senior cloud FinOps engineer. Given the alert metadata "
        "below, write a 3-4 sentence hypothesis about what caused the cost "
        "anomaly. Be specific about what to check next (CloudTrail events, "
        "deploy windows, autoscaling rules, etc.). Avoid generic advice.\n\n"
        f"Alert: {alert}\n\n"
        f"Attribution hint: {attribution}\n\n"
        f"Last 7 days of CUR for this service:\n"
        f"{cur_window.to_string(index=False) if not cur_window.empty else '(no data)'}\n"
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )
        return Explanation(
            text=text,
            source="anthropic",
            model=model,
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
        )
    except Exception:
        return None


def explain_alert(
    alert: pd.Series | dict,
    attribution: pd.Series | dict,
    cur_df: pd.DataFrame,
    *,
    model: str = "claude-haiku-4-5",
    force_template: bool = False,
) -> Explanation:
    """Generate a natural-language root-cause hypothesis for an alert.

    Args:
        alert: a row from build_alerts (or equivalent dict).
        attribution: matching row from attribute(...), or its dict form.
        cur_df: full CUR — used to extract the 7-day trailing window.
        model: Anthropic model id. Defaults to claude-haiku-4-5.
        force_template: skip the API even if the key is set (testing).
    """
    a = dict(alert) if not isinstance(alert, dict) else alert
    attr = dict(attribution) if not isinstance(attribution, dict) else attribution

    key = _cache_key(a, attr)
    if key in _CACHE:
        cached = _CACHE[key]
        return Explanation(
            text=cached.text, source="cache", model=cached.model,
            input_tokens=cached.input_tokens, output_tokens=cached.output_tokens,
        )

    # Build the trailing 7-day window for context.
    anom_date = pd.to_datetime(a.get("date"))
    service = a.get("service", "")
    cur_window = pd.DataFrame()
    if not cur_df.empty and pd.notna(anom_date):
        c = cur_df.copy()
        c["date"] = pd.to_datetime(c["date"])
        cur_window = c[
            (c["service"] == service)
            & (c["date"] >= anom_date - pd.Timedelta(days=7))
            & (c["date"] <= anom_date)
        ].copy()
        cur_window = cur_window.groupby("date", as_index=False)["cost"].sum()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not force_template and api_key:
        live = _call_anthropic(a, attr, cur_window, model=model, api_key=api_key)
        if live is not None:
            _CACHE[key] = live
            return live

    text = _template_explanation(a, attr, cur_window)
    result = Explanation(text=text, source="template", model="deterministic")
    _CACHE[key] = result
    return result


def clear_cache() -> None:
    _CACHE.clear()
