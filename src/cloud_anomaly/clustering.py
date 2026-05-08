"""Cluster similar alerts together so reviewers see "incidents" not noise.

A real FinOps team rarely cares about individual alert rows — they care
about *incidents*: "the 4 alerts on April 14-17 are all the same RDS
drift". This module groups alerts whose features are close in
multi-dimensional space (date proximity, service identity, severity,
anomaly type) into clusters and emits a one-line incident summary.

Implementation: scikit-learn DBSCAN over a hand-crafted feature vector,
deliberately interpretable (no UMAP, no embeddings).
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_alerts(
    alerts: pd.DataFrame,
    *,
    eps: float = 0.85,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Group similar alerts into "incidents".

    Returns the alerts frame with two new columns:
      - ``incident_id`` : -1 for un-clustered (singleton) alerts, otherwise
        the cluster id starting at 0.
      - ``incident_size`` : how many alerts share the same incident_id.

    Tunable knobs:
      * ``eps`` — DBSCAN density radius. Smaller = more incidents.
      * ``min_samples`` — minimum alerts to form an incident (2 is the
        smallest meaningful "this is a pattern, not noise" threshold).
    """
    if alerts.empty:
        out = alerts.copy()
        out["incident_id"] = pd.Series(dtype=int)
        out["incident_size"] = pd.Series(dtype=int)
        return out

    df = alerts.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Feature engineering — every column normalized.
    features = pd.DataFrame({
        # Day-of-history (so close-in-time alerts cluster).
        "day_idx": (df["date"] - df["date"].min()).dt.days.astype(float),
        # Severity score directly (0–1).
        "severity": df["severity_score"].astype(float),
        # Categorical → integer codes.
        "service_code": df["service"].astype("category").cat.codes.astype(float),
    })
    if "detector" in df.columns:
        features["detector_code"] = df["detector"].astype("category").cat.codes.astype(float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled)

    df["incident_id"] = labels
    sizes = pd.Series(labels).value_counts()
    df["incident_size"] = pd.Series(labels).map(sizes).values
    return df


def summarize_incidents(clustered: pd.DataFrame) -> pd.DataFrame:
    """One row per incident with a short human-readable description."""
    if clustered.empty or "incident_id" not in clustered.columns:
        return pd.DataFrame(
            columns=[
                "incident_id", "n_alerts", "first_date", "last_date",
                "services", "max_severity", "summary",
            ]
        )

    incidents = clustered[clustered["incident_id"] >= 0]
    if incidents.empty:
        return pd.DataFrame(
            columns=[
                "incident_id", "n_alerts", "first_date", "last_date",
                "services", "max_severity", "summary",
            ]
        )

    rows = []
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    for incident_id, group in incidents.groupby("incident_id"):
        n = len(group)
        first = group["date"].min()
        last = group["date"].max()
        services = sorted(group["service"].unique())
        services_str = ", ".join(services[:3]) + (f" (+{len(services)-3} more)" if len(services) > 3 else "")
        max_sev = max(group["severity"], key=lambda s: severity_rank.get(s, 0))
        days = (last - first).days + 1

        if first == last:
            timing = first.strftime("%Y-%m-%d")
        else:
            timing = f"{first:%Y-%m-%d} → {last:%Y-%m-%d} ({days} days)"

        summary = (
            f"{n} alerts on {services_str} during {timing}. "
            f"Max severity = {max_sev}."
        )
        rows.append({
            "incident_id": int(incident_id),
            "n_alerts": int(n),
            "first_date": first,
            "last_date": last,
            "services": services_str,
            "max_severity": max_sev,
            "summary": summary,
        })

    return pd.DataFrame(rows).sort_values("n_alerts", ascending=False).reset_index(drop=True)
