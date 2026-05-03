"""Streamlit dashboard for the cloud cost anomaly detector.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.alerts import build_alerts  # noqa: E402
from cloud_anomaly.config import RAW_DIR  # noqa: E402
from cloud_anomaly.detectors import DETECTORS  # noqa: E402
from cloud_anomaly.evaluation import compare_detectors  # noqa: E402
from cloud_anomaly.preprocessing import aggregate_by_service, aggregate_daily, load_cur  # noqa: E402
from cloud_anomaly.synthetic_data import generate  # noqa: E402


st.set_page_config(
    page_title="Cloud Cost Anomaly Detector",
    page_icon="☁️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load(regenerate: bool, n_days: int, seed: int):
    if regenerate or not (RAW_DIR / "cur_synthetic.parquet").exists():
        cur_df, labels_df, _ = generate(n_days=n_days, seed=seed)
    else:
        cur_df = load_cur()
        labels_df = pd.read_csv(RAW_DIR / "ground_truth_labels.csv", parse_dates=["date"])
    long = aggregate_by_service(cur_df)
    daily = aggregate_daily(cur_df)
    return cur_df, labels_df, long, daily


@st.cache_data(show_spinner=False)
def _run_detectors(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: fn(long) for name, fn in DETECTORS.items()}


def main() -> None:
    st.title("☁️ Automated Cloud Cost Anomaly Detector")
    st.caption("Project 13 · Cloud Computing · Spring 2025–2026")

    with st.sidebar:
        st.header("⚙️ Configuration")
        regenerate = st.checkbox("Regenerate synthetic data", value=False)
        n_days = st.slider("Days of history", 30, 180, 90, step=15)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        st.markdown("---")
        st.subheader("Detector")
        detector_name = st.selectbox(
            "Active detector",
            options=list(DETECTORS.keys()),
            format_func=lambda x: {
                "zscore": "Z-Score (baseline)",
                "stl": "STL Decomposition",
                "iforest": "Isolation Forest",
            }[x],
        )
        severity_filter = st.multiselect(
            "Severity filter",
            options=["LOW", "MEDIUM", "HIGH"],
            default=["MEDIUM", "HIGH"],
        )

    cur_df, labels_df, long, daily = _load(regenerate, n_days, int(seed))
    detectors = _run_detectors(long)
    detections = detectors[detector_name]
    alerts = build_alerts(detections, detector_name, dataset_days=long["date"].nunique())

    # Tag every (date, service) row with TP / FP / FN / TN against ground truth.
    # Used by the cost-trend overlay and the alert-log "ground_truth" column.
    truth = labels_df[["date", "service", "is_anomaly"]].rename(
        columns={"is_anomaly": "_truth"}
    )
    tagged = detections.merge(truth, on=["date", "service"], how="left")
    tagged["_truth"] = tagged["_truth"].fillna(False).astype(bool)
    tagged["outcome"] = "TN"
    tagged.loc[tagged["is_anomaly"] & tagged["_truth"], "outcome"] = "TP"
    tagged.loc[tagged["is_anomaly"] & ~tagged["_truth"], "outcome"] = "FP"
    tagged.loc[~tagged["is_anomaly"] & tagged["_truth"], "outcome"] = "FN"

    # KPIs.
    total_spend = cur_df["cost"].sum()
    n_alerts = int(detections["is_anomaly"].sum())
    n_tp = int((tagged["outcome"] == "TP").sum())
    n_fp = int((tagged["outcome"] == "FP").sum())
    n_fn = int((tagged["outcome"] == "FN").sum())
    tp_precision = n_tp / max(n_tp + n_fp, 1)
    n_high = int((alerts["severity"] == "HIGH").sum()) if not alerts.empty else 0
    n_services = cur_df["service"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total spend", f"${total_spend:,.0f}")
    k2.metric("Services", n_services)
    k3.metric("Anomalies flagged", n_alerts)
    k4.metric("HIGH-severity alerts", n_high)
    k5.metric(
        "True-positive rate",
        f"{tp_precision:.0%}",
        help=f"TP={n_tp} · FP={n_fp} · FN={n_fn} (vs. ground truth)",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Cost trend", "🚨 Alert log", "📊 Detector comparison", "🔍 Raw data"]
    )

    with tab1:
        st.subheader("Daily total cloud spend")
        # Color-coded ground-truth overlay: TP=green, FP=red, FN=amber.
        # A given day rolls up to the worst outcome across services so the
        # daily-total chart still shows useful signal.
        priority = {"FN": 3, "FP": 2, "TP": 1, "TN": 0}
        per_day = (
            tagged.assign(_rank=tagged["outcome"].map(priority))
            .sort_values("_rank", ascending=False)
            .drop_duplicates("date", keep="first")[["date", "outcome"]]
        )
        daily_outcome = daily.merge(per_day, on="date", how="left")
        outcome_styles = {
            "TP": ("True positive", "#16a34a", "circle"),
            "FP": ("False positive", "#dc2626", "x"),
            "FN": ("Missed (false negative)", "#f59e0b", "triangle-up"),
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["cost"], name="Daily cost", mode="lines"))
        for code, (label, color, symbol) in outcome_styles.items():
            sub = daily_outcome[daily_outcome["outcome"] == code]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["cost"],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=11, symbol=symbol, line=dict(width=1, color="white")),
                )
            )
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Cost ($)", height=420, hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🟢 True positive · 🔴 False positive · 🟡 Missed anomaly (false negative). "
            "Color reflects the worst outcome across services on each day."
        )

        st.subheader("Per-service breakdown")
        per_service = px.line(
            long, x="date", y="cost", color="service", height=380,
            title="Daily cost by service",
        )
        st.plotly_chart(per_service, use_container_width=True)

    with tab2:
        st.subheader("Alerts")
        if alerts.empty:
            st.info("No anomalies flagged with current settings.")
        else:
            outcome_lookup = tagged.set_index(["date", "service"])["outcome"]
            enriched = alerts.copy()
            enriched["ground_truth"] = [
                "TP" if outcome_lookup.get((d, s)) == "TP" else "FP"
                for d, s in zip(enriched["date"], enriched["service"])
            ]
            view = enriched[enriched["severity"].isin(severity_filter)].copy()
            view["date"] = view["date"].dt.strftime("%Y-%m-%d")
            view["severity_score"] = view["severity_score"].round(3)
            view["score"] = view["score"].round(3)
            view["cost"] = view["cost"].round(2)
            st.dataframe(view, use_container_width=True, hide_index=True)
            n_view_tp = int((view["ground_truth"] == "TP").sum())
            n_view = len(view)
            if n_view:
                st.caption(
                    f"Filtered alerts: {n_view_tp} / {n_view} are true positives "
                    f"({n_view_tp / n_view:.0%} precision in the current view)."
                )
            st.download_button(
                "⬇️ Download alerts (CSV)",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name=f"alerts_{detector_name}.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Precision / Recall by anomaly type")
        comparison = compare_detectors(detectors, labels_df)
        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True,
        )
        chart = px.bar(
            comparison[comparison["anomaly_type"] != "OVERALL"],
            x="anomaly_type",
            y="f1",
            color="detector",
            barmode="group",
            title="F1 score by anomaly type",
            height=380,
        )
        st.plotly_chart(chart, use_container_width=True)

    with tab4:
        st.subheader("Synthetic CUR rows (sample)")
        st.dataframe(cur_df.head(200), use_container_width=True, hide_index=True)
        st.subheader("Ground-truth labels")
        gt = labels_df[labels_df["is_anomaly"]].copy()
        gt["date"] = gt["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(gt, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
