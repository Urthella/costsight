"""Root-cause attribution for flagged anomalies.

For each (date, service) flagged by a detector, decompose the spend
along available CUR dimensions (region, usage_type) and report the
contributor that drove the cost above its 14-day rolling baseline.

Output schema:
    date, service, total_cost, baseline_cost, delta, severity,
    top_dimension, top_value, top_value_share, top_value_delta, summary

The ``summary`` column is a short human-readable string that the
dashboard surfaces directly — e.g.

    "EC2 spend on 2025-03-19 is $1,247 (+312% vs 14-day baseline);
     us-east-1 region drove 72% of the increase."
"""
from __future__ import annotations

import pandas as pd


_ATTRIBUTION_DIMS = ["region", "usage_type"]


def attribute(
    cur_df: pd.DataFrame,
    alerts: pd.DataFrame,
    baseline_window: int = 14,
) -> pd.DataFrame:
    """Compute a one-line root-cause hint for every alert.

    Args:
        cur_df: raw CUR rows with date, service, region, usage_type, cost.
        alerts: alert log produced by ``alerts.build_alerts``.
        baseline_window: rolling window in days used as the "normal" reference.
    """
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "date", "service", "total_cost", "baseline_cost", "delta",
                "severity", "top_dimension", "top_value", "top_value_share",
                "top_value_delta", "summary",
            ]
        )

    cur = cur_df.copy()
    cur["date"] = pd.to_datetime(cur["date"])

    # Daily total per service (used to compute the baseline).
    daily_service = (
        cur.groupby(["date", "service"], as_index=False)["cost"].sum()
        .sort_values(["service", "date"])
    )
    daily_service["baseline"] = (
        daily_service.groupby("service")["cost"]
        .transform(lambda s: s.rolling(baseline_window, min_periods=3).mean().shift(1))
    )

    rows = []
    for _, alert in alerts.iterrows():
        anom_date = pd.to_datetime(alert["date"])
        service = alert["service"]
        same = cur[(cur["date"] == anom_date) & (cur["service"] == service)]
        if same.empty:
            continue
        total = float(same["cost"].sum())

        baseline_row = daily_service[
            (daily_service["date"] == anom_date) & (daily_service["service"] == service)
        ]
        baseline = float(baseline_row["baseline"].iloc[0]) if not baseline_row.empty else 0.0
        delta = total - baseline

        # Find which (dimension, value) contributed most to the delta.
        best = {"dim": None, "value": None, "share": 0.0, "delta": 0.0}
        for dim in _ATTRIBUTION_DIMS:
            if dim not in cur.columns:
                continue
            grouped = same.groupby(dim, as_index=False)["cost"].sum()
            if grouped.empty:
                continue
            # Per-value baseline: mean over the trailing window for the
            # same (service, dim_value) combination.
            window_start = anom_date - pd.Timedelta(days=baseline_window)
            window_data = cur[
                (cur["service"] == service)
                & (cur["date"] >= window_start)
                & (cur["date"] < anom_date)
            ]
            if not window_data.empty:
                baseline_per_value = (
                    window_data.groupby(dim)["cost"].sum() / max(baseline_window, 1)
                ).rename("baseline")
            else:
                baseline_per_value = pd.Series(dtype=float, name="baseline")
            grouped = grouped.merge(
                baseline_per_value, left_on=dim, right_index=True, how="left"
            )
            grouped["baseline"] = grouped["baseline"].fillna(0.0)
            grouped["delta"] = grouped["cost"] - grouped["baseline"]

            top = grouped.sort_values("delta", ascending=False).iloc[0]
            value_delta = float(top["delta"])
            share = value_delta / delta if delta else 0.0

            if value_delta > best["delta"]:
                best = {
                    "dim": dim,
                    "value": str(top[dim]),
                    "share": share,
                    "delta": value_delta,
                }

        pct = (delta / baseline * 100.0) if baseline else 0.0
        if best["dim"] is None:
            summary = (
                f"{service} spend on {anom_date:%Y-%m-%d} is "
                f"${total:,.0f} (delta ${delta:+,.0f} vs baseline)."
            )
        else:
            summary = (
                f"{service} spend on {anom_date:%Y-%m-%d} is ${total:,.0f} "
                f"({pct:+.0f}% vs {baseline_window}-day baseline); "
                f"{best['value']} {best['dim']} drove "
                f"{max(0.0, min(best['share'], 1.0)) * 100:.0f}% of the increase."
            )

        rows.append(
            {
                "date": anom_date,
                "service": service,
                "total_cost": round(total, 2),
                "baseline_cost": round(baseline, 2),
                "delta": round(delta, 2),
                "severity": alert.get("severity", ""),
                "top_dimension": best["dim"] or "",
                "top_value": best["value"] or "",
                "top_value_share": round(max(0.0, min(best["share"], 1.0)), 4),
                "top_value_delta": round(best["delta"], 2),
                "summary": summary,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["severity", "delta"], ascending=[True, False]
    ).reset_index(drop=True)
