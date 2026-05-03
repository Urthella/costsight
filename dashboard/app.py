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
from cloud_anomaly.attribution import attribute  # noqa: E402
from cloud_anomaly.config import RAW_DIR  # noqa: E402
from cloud_anomaly.detectors import DETECTORS  # noqa: E402
from cloud_anomaly.evaluation import (  # noqa: E402
    compare_detectors,
    cost_saved_estimate,
    evaluate_alerts,
    evaluate_by_type,
    time_to_detect,
)
from cloud_anomaly.detectors.zscore import detect as zscore_detect_raw  # noqa: E402
from cloud_anomaly.detectors.stl import detect as stl_detect_raw  # noqa: E402
from cloud_anomaly.detectors.iforest import detect as iforest_detect_raw  # noqa: E402
from cloud_anomaly.preprocessing import aggregate_by_service, aggregate_daily, load_cur  # noqa: E402
from cloud_anomaly.synthetic_data import generate  # noqa: E402
from cloud_anomaly.theoretical_scores import (  # noqa: E402
    INTERPRETABILITY_QUALITATIVE,
    RADAR_AXES,
    THEORETICAL_SCORES,
)


st.set_page_config(
    page_title="Cloud Cost Anomaly Detector",
    page_icon="☁️",
    layout="wide",
)

DETECTOR_LABELS = {
    "zscore": "Z-Score (baseline)",
    "stl": "STL Decomposition",
    "iforest": "Isolation Forest",
    "ensemble": "Ensemble (≥2 vote)",
}

DETECTOR_STYLES = {
    "zscore": {"color": "#3B82F6", "symbol": "circle"},
    "stl": {"color": "#F59E0B", "symbol": "square"},
    "iforest": {"color": "#A855F7", "symbol": "triangle-up"},
    "ensemble": {"color": "#10B981", "symbol": "star"},
}


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


@st.cache_data(show_spinner=False)
def _measure_speed(long: pd.DataFrame) -> dict[str, float]:
    """Time each detector once on the live dataset; normalize to [0, 1].

    Higher = faster. Median over 3 runs (cheap; first call also primes caches).
    """
    import time

    raw = {}
    for name, fn in DETECTORS.items():
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn(long)
            runs.append(time.perf_counter() - t0)
        runs.sort()
        raw[name] = runs[len(runs) // 2]
    fastest = min(raw.values())
    return {name: fastest / t for name, t in raw.items()}


def _build_radar_df(empirical_speed: dict[str, float]) -> pd.DataFrame:
    """Assemble theoretical-vs-empirical scores per detector across 5 axes."""
    benchmark_path = ROOT / "outputs" / "benchmark_summary.csv"
    if benchmark_path.exists():
        bench = pd.read_csv(benchmark_path)
    else:
        bench = pd.DataFrame(columns=["detector", "anomaly_type", "f1_mean"])

    type_axis = {
        "point_spike": "Point Spike",
        "level_shift": "Level Shift",
        "gradual_drift": "Gradual Drift",
    }

    rows = []
    for det in DETECTORS.keys():
        for kind, axis in type_axis.items():
            empirical = bench[(bench["detector"] == det) & (bench["anomaly_type"] == kind)]
            f1 = float(empirical["f1_mean"].iloc[0]) if not empirical.empty else 0.0
            rows.append({"detector": det, "axis": axis, "kind": "Empirical", "value": f1})
        rows.append({
            "detector": det, "axis": "Speed", "kind": "Empirical",
            "value": empirical_speed.get(det, 0.0),
        })
        rows.append({
            "detector": det, "axis": "Interpretability", "kind": "Empirical",
            "value": INTERPRETABILITY_QUALITATIVE.get(det, 0.0),
        })
        for axis in RADAR_AXES:
            rows.append({
                "detector": det, "axis": axis, "kind": "Theoretical",
                "value": THEORETICAL_SCORES[det][axis],
            })
    return pd.DataFrame(rows)


def _build_all_alerts(detections_by_name: dict[str, pd.DataFrame], dataset_days: int) -> pd.DataFrame:
    frames = [
        build_alerts(detections, name, dataset_days=dataset_days)
        for name, detections in detections_by_name.items()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _union_alert_view(all_alerts: pd.DataFrame) -> pd.DataFrame:
    """Union per-(date,service) across detectors with a flagged_by list."""
    if all_alerts.empty:
        return all_alerts
    grouped = (
        all_alerts.groupby(["date", "service"], as_index=False)
        .agg(
            cost=("cost", "max"),
            severity_score=("severity_score", "max"),
            severity=("severity", lambda s: max(s, key=lambda v: ["LOW", "MEDIUM", "HIGH"].index(v))),
            flagged_by=("detector", lambda s: ", ".join(sorted(set(s)))),
            n_detectors=("detector", "nunique"),
        )
        .sort_values(["severity_score", "date"], ascending=[False, True])
    )
    return grouped


def main() -> None:
    st.title("☁️ Automated Cloud Cost Anomaly Detector")
    st.caption("Project 13 · Cloud Computing · Spring 2025–2026")

    with st.sidebar:
        st.header("⚙️ Configuration")
        regenerate = st.checkbox("Regenerate synthetic data", value=False)
        n_days = st.slider("Days of history", 30, 180, 90, step=15)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        st.markdown("---")
        st.subheader("Detectors")
        active_detectors = st.multiselect(
            "Active detectors (overlay)",
            options=list(DETECTORS.keys()),
            default=list(DETECTORS.keys()),
            format_func=lambda x: DETECTOR_LABELS[x],
        )
        severity_filter = st.multiselect(
            "Severity filter",
            options=["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
        )

    if not active_detectors:
        st.warning("Select at least one detector in the sidebar.")
        return
    if not severity_filter:
        st.warning("Select at least one severity band in the sidebar.")
        return

    cur_df, labels_df, long, daily = _load(regenerate, n_days, int(seed))
    detectors_all = _run_detectors(long)
    detections_by_name = {k: detectors_all[k] for k in active_detectors}

    dataset_days = long["date"].nunique()
    alerts_by_name = {
        name: build_alerts(det, name, dataset_days=dataset_days)
        for name, det in detections_by_name.items()
    }
    all_alerts = _build_all_alerts(detections_by_name, dataset_days)

    if all_alerts.empty:
        filtered_alerts = all_alerts
    else:
        filtered_alerts = all_alerts[all_alerts["severity"].isin(severity_filter)].copy()

    # KPIs.
    total_spend = cur_df["cost"].sum()
    n_services = cur_df["service"].nunique()
    per_detector_counts = {
        name: int(det["is_anomaly"].sum()) for name, det in detections_by_name.items()
    }
    counts_str = " · ".join(
        f"{DETECTOR_LABELS[name].split(' ')[0]}: {n}"
        for name, n in per_detector_counts.items()
    )
    n_high = int((filtered_alerts["severity"] == "HIGH").sum()) if not filtered_alerts.empty else 0
    n_consensus = (
        int((filtered_alerts.get("flagged_by", pd.Series(dtype=str)).str.count(",") >= 1).sum())
        if not filtered_alerts.empty else 0
    )

    # Cost-saved estimate using the best-performing active detector by F1.
    primary_for_savings = None
    best_f1 = -1.0
    for name in active_detectors:
        f1 = float(evaluate_by_type(detections_by_name[name], labels_df).iloc[-1]["f1"])
        if f1 > best_f1:
            best_f1 = f1
            primary_for_savings = name
    saved_info = (
        cost_saved_estimate(cur_df, detections_by_name[primary_for_savings], labels_df)
        if primary_for_savings else {"saved": 0.0, "total_anomaly_cost": 0.0, "ratio": 0.0}
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total spend", f"${total_spend:,.0f}")
    k2.metric("Services", n_services)
    k3.metric("Anomalies (per detector)", counts_str if counts_str else "—")
    k4.metric("Consensus alerts", n_consensus, help="Alerts flagged by ≥2 detectors (severity-filtered)")
    k5.metric(
        f"$ savable (best: {DETECTOR_LABELS.get(primary_for_savings, '—').split(' ')[0]})",
        f"${saved_info['saved']:,.0f}",
        delta=f"{saved_info['ratio']*100:.0f}% of leak",
        help=(
            "If FinOps acted 1 day after the first detection, this much excess "
            "spend would have been avoided. Total leak in window = "
            f"${saved_info['total_anomaly_cost']:,.0f}."
        ),
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📈 Cost trend", "🚨 Alert log", "🔎 Root-cause",
            "📊 Detector comparison", "🔬 Lab", "🎬 Replay", "🗂️ Raw data",
        ]
    )

    # Severity-filtered detection sets per detector, used for chart markers.
    flagged_dates_per_detector: dict[str, set] = {}
    for name in active_detectors:
        if alerts_by_name[name].empty:
            flagged_dates_per_detector[name] = set()
            continue
        kept = alerts_by_name[name][alerts_by_name[name]["severity"].isin(severity_filter)]
        flagged_dates_per_detector[name] = set(zip(kept["date"], kept["service"]))

    with tab1:
        st.subheader("Daily total cloud spend")
        st.caption(
            "Green shading = ground-truth anomaly window. Markers = detector flags "
            "(severity-filtered). The gap between green shading and markers tells "
            "you what each detector caught vs missed."
        )
        show_truth = st.checkbox("Overlay ground-truth anomaly windows", value=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cost"], name="Daily cost",
            mode="lines", line=dict(color="#94A3B8", width=2),
        ))
        if show_truth and not labels_df.empty:
            truth_dates = labels_df.loc[labels_df["is_anomaly"], "date"].unique()
            for tdate in truth_dates:
                fig.add_vrect(
                    x0=pd.Timestamp(tdate) - pd.Timedelta(hours=12),
                    x1=pd.Timestamp(tdate) + pd.Timedelta(hours=12),
                    fillcolor="#10B981", opacity=0.18, layer="below", line_width=0,
                )
        for name in active_detectors:
            flagged = flagged_dates_per_detector[name]
            if not flagged:
                continue
            flagged_dates = {d for d, _ in flagged}
            day_view = daily[daily["date"].isin(flagged_dates)]
            style = DETECTOR_STYLES[name]
            fig.add_trace(go.Scatter(
                x=day_view["date"], y=day_view["cost"],
                mode="markers", name=DETECTOR_LABELS[name],
                marker=dict(color=style["color"], size=11, symbol=style["symbol"], line=dict(width=1, color="white")),
            ))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Cost ($)", height=420,
            hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-service breakdown")
        st.caption("This is where anomalies actually live — drift and level shifts are visible per service.")
        per_service = px.line(
            long, x="date", y="cost", color="service", height=420,
        )
        # Overlay per-(date, service) markers per detector on top of the per-service lines.
        for name in active_detectors:
            flagged = flagged_dates_per_detector[name]
            if not flagged:
                continue
            flagged_df = pd.DataFrame(list(flagged), columns=["date", "service"])
            merged = flagged_df.merge(long, on=["date", "service"], how="left")
            style = DETECTOR_STYLES[name]
            per_service.add_trace(go.Scatter(
                x=merged["date"], y=merged["cost"],
                mode="markers", name=DETECTOR_LABELS[name],
                marker=dict(color=style["color"], size=10, symbol=style["symbol"],
                            line=dict(width=1, color="white")),
                showlegend=True,
            ))
        per_service.update_layout(
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(per_service, use_container_width=True)

    with tab2:
        st.subheader("Alerts (union across active detectors)")
        st.caption(
            "Each row is a (date, service) pair. `flagged_by` shows which detectors "
            "agreed. Consensus alerts (≥2 detectors) are the most actionable."
        )
        if filtered_alerts.empty:
            st.info("No anomalies flagged with current settings.")
        else:
            view = _union_alert_view(filtered_alerts)
            view["date"] = view["date"].dt.strftime("%Y-%m-%d")
            view["severity_score"] = view["severity_score"].round(3)
            view["cost"] = view["cost"].round(2)
            display = view[[
                "date", "service", "severity", "severity_score",
                "cost", "flagged_by", "n_detectors",
            ]]
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download alerts (CSV)",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name="alerts_union.csv",
                mime="text/csv",
            )

            with st.expander("🔍 Why did each alert fire? — pick one"):
                pick_options = [f"{r['date']} · {r['service']}" for _, r in display.iterrows()]
                if pick_options:
                    pick = st.selectbox("Alert", options=pick_options, key="why_pick")
                    pick_idx = pick_options.index(pick)
                    row = display.iloc[pick_idx]
                    sev_score = float(row["severity_score"])
                    detector_msgs = {
                        "zscore": "rolling 14-day z-score exceeded the |z|≥3 threshold",
                        "stl": "STL residual / trend deviation exceeded its threshold",
                        "iforest": "Isolation Forest assigned an anomaly score above 0.55",
                        "ensemble": "≥2 base detectors agreed",
                    }
                    reasons = ", ".join(
                        detector_msgs.get(d.strip(), d.strip()) for d in row["flagged_by"].split(",")
                    )
                    st.markdown(
                        f"**{row['service']} on {row['date']}** — cost **${row['cost']:,.2f}**, "
                        f"severity **{row['severity']}** (score = {sev_score:.3f}).\n\n"
                        f"Flagged because: {reasons}.\n\n"
                        f"Severity blends detector deviation × duration of the flagged "
                        f"run × dollar impact. Severity {sev_score:.3f} ≥ "
                        f"{ '0.66 → HIGH' if sev_score >= 0.66 else '0.33 → MEDIUM' if sev_score >= 0.33 else 'below 0.33 → LOW' }."
                    )

    with tab3:
        st.subheader("Root-cause attribution")
        st.caption(
            "For every flagged (date, service), we decompose the spend along "
            "region and usage_type and report the dimension that drove the "
            "increase the most vs. the 14-day rolling baseline. Computed per "
            "active detector — switch detectors below to see how the hint shifts."
        )
        attr_detector = st.radio(
            "Attribution source",
            options=active_detectors,
            format_func=lambda x: DETECTOR_LABELS[x],
            horizontal=True,
        )
        alerts_for_attr = alerts_by_name[attr_detector]
        attribution_df = attribute(cur_df, alerts_for_attr)
        if attribution_df.empty:
            st.info("No alerts → no attributions to compute.")
        else:
            view = attribution_df[attribution_df["severity"].isin(severity_filter)].copy()
            if view.empty:
                st.info("No attributions match the active severity filter.")
            else:
                view["date"] = view["date"].dt.strftime("%Y-%m-%d")
                view["top_value_share"] = (view["top_value_share"] * 100).round(0).astype(int).astype(str) + "%"
                display = view[[
                    "date", "service", "severity", "summary",
                    "top_dimension", "top_value", "top_value_share",
                    "total_cost", "baseline_cost", "delta",
                ]]
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download attributions (CSV)",
                    data=view.drop(columns=["top_value_share"]).to_csv(index=False).encode("utf-8"),
                    file_name=f"attribution_{attr_detector}.csv",
                    mime="text/csv",
                )

    with tab4:
        st.subheader("Precision / Recall by anomaly type")
        comparison = compare_detectors(detectors_all, labels_df)
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

        st.markdown("---")
        st.subheader("Theoretical vs. Empirical — pentagon comparison")
        st.caption(
            "Dashed = a-priori expectation from the proposal (textbook reasoning). "
            "Solid = measured: F1 from the 25-seed benchmark for the three anomaly "
            "axes; live timing for Speed; qualitative for Interpretability. "
            "Where dashed and solid diverge is where the project's empirical work "
            "added information beyond textbook predictions."
        )
        speed_norm = _measure_speed(long)
        radar_df = _build_radar_df(speed_norm)
        cols = st.columns(3)
        for col, det in zip(cols, DETECTORS.keys()):
            sub = radar_df[radar_df["detector"] == det]
            theo = sub[sub["kind"] == "Theoretical"].set_index("axis").loc[list(RADAR_AXES)]
            emp = sub[sub["kind"] == "Empirical"].set_index("axis").loc[list(RADAR_AXES)]
            color = DETECTOR_STYLES[det]["color"]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=list(theo["value"]) + [theo["value"].iloc[0]],
                theta=list(RADAR_AXES) + [RADAR_AXES[0]],
                name="Theoretical",
                line=dict(color=color, dash="dash", width=2),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
            ))
            radar.add_trace(go.Scatterpolar(
                r=list(emp["value"]) + [emp["value"].iloc[0]],
                theta=list(RADAR_AXES) + [RADAR_AXES[0]],
                name="Empirical",
                line=dict(color=color, width=3),
                fill="toself",
                fillcolor=color,
                opacity=0.45,
            ))
            radar.update_layout(
                title=DETECTOR_LABELS[det],
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                legend=dict(orientation="h", y=-0.15),
                height=380,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            col.plotly_chart(radar, use_container_width=True)

        with st.expander("How to read the pentagon"):
            st.markdown(
                "- **Point Spike / Level Shift / Gradual Drift** — F1 score on each "
                "anomaly type. Empirical = mean of 25 seeds.\n"
                "- **Speed** — inverse runtime, normalized so the fastest detector = 1.0.\n"
                "- **Interpretability** — qualitative score (how easily a FinOps "
                "engineer can explain *why* a day was flagged).\n"
                "- A solid pentagon **inside** the dashed line = we over-estimated "
                "that detector. A solid pentagon **outside** the dashed line = it "
                "performed better than the proposal predicted."
            )

        st.markdown("---")
        st.subheader("Time-to-detect by anomaly type")
        st.caption(
            "How many days after an anomaly *starts* does each detector first flag "
            "it? Lower is better. NaN = never detected within the window — those "
            "are missed anomalies."
        )
        ttd_rows = []
        for name in active_detectors:
            ttd = time_to_detect(detections_by_name[name], labels_df)
            if ttd.empty:
                continue
            ttd["detector"] = name
            ttd_rows.append(ttd)
        if ttd_rows:
            ttd_all = pd.concat(ttd_rows, ignore_index=True)
            ttd_summary = (
                ttd_all.groupby(["detector", "anomaly_type"])
                .agg(
                    median_days=("days_to_detect", "median"),
                    detected=("days_to_detect", lambda s: int(s.notna().sum())),
                    total=("days_to_detect", "size"),
                )
                .reset_index()
            )
            ttd_summary["miss_rate"] = (
                1 - ttd_summary["detected"] / ttd_summary["total"]
            ).round(2)
            ttd_summary["detector"] = ttd_summary["detector"].map(DETECTOR_LABELS).fillna(ttd_summary["detector"])
            ttd_chart = px.bar(
                ttd_summary,
                x="anomaly_type", y="median_days", color="detector",
                barmode="group", height=360,
                title="Median days to first detection",
                labels={"median_days": "Median days", "anomaly_type": "Anomaly type"},
            )
            st.plotly_chart(ttd_chart, use_container_width=True)
            st.dataframe(ttd_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No TTD data — check that ground-truth labels are loaded.")

        st.markdown("---")
        st.subheader("ROC — true-positive vs false-positive at every score threshold")
        st.caption(
            "We sweep each detector's score from low to high and plot the resulting "
            "TPR/FPR. The closer a curve hugs the top-left corner, the better the "
            "detector separates anomalies from normal days at *any* threshold."
        )
        roc_fig = go.Figure()
        for name in active_detectors:
            det = detections_by_name[name]
            score_col = "score" if "score" in det.columns else None
            if score_col is None:
                continue
            scored = det.merge(
                labels_df[["date", "service", "is_anomaly"]].rename(columns={"is_anomaly": "_truth"}),
                on=["date", "service"], how="left",
            )
            scored["_truth"] = scored["_truth"].fillna(False).astype(bool)
            scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
            tp = (scored["_truth"]).cumsum()
            fp = (~scored["_truth"]).cumsum()
            P = max(int(scored["_truth"].sum()), 1)
            N = max(int((~scored["_truth"]).sum()), 1)
            tpr = tp / P
            fpr = fp / N
            color = DETECTOR_STYLES[name]["color"]
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=DETECTOR_LABELS[name], line=dict(color=color, width=2),
            ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#555", dash="dash", width=1),
        ))
        roc_fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with tab5:
        st.subheader("🔬 Threshold sensitivity playground")
        st.caption(
            "Move the sliders below — each detector re-runs in real time on the "
            "current dataset and the P/R/F1 table updates live. The defaults match "
            "the values used in the 25-seed benchmark."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Z-Score**")
            zs_window = st.slider("Window (days)", 7, 30, 14, key="zs_w")
            zs_thresh = st.slider("|z| threshold", 1.5, 5.0, 3.0, 0.1, key="zs_t")
        with c2:
            st.markdown("**STL Decomposition**")
            stl_period = st.slider("Seasonal period", 5, 14, 7, key="stl_p")
            stl_thresh = st.slider("|residual/σ| threshold", 1.5, 5.0, 3.0, 0.1, key="stl_t")
        with c3:
            st.markdown("**Isolation Forest**")
            if_contamination = st.slider("Contamination", 0.02, 0.20, 0.08, 0.01, key="if_c")
            if_score = st.slider("Score threshold", 0.40, 0.80, 0.55, 0.01, key="if_s")

        custom = {
            "zscore": zscore_detect_raw(long, window=zs_window, threshold=zs_thresh),
            "stl": stl_detect_raw(long, period=stl_period, threshold=stl_thresh),
            "iforest": iforest_detect_raw(
                long, contamination=if_contamination, score_threshold=if_score,
            ),
        }
        custom_compare = compare_detectors(custom, labels_df)
        st.dataframe(custom_compare.round(3), use_container_width=True, hide_index=True)

        custom_overall = custom_compare[custom_compare["anomaly_type"] == "OVERALL"]
        chart = px.bar(
            custom_overall, x="detector", y="f1", color="detector",
            title="Overall F1 with custom thresholds",
            range_y=[0, 1.0], height=320,
        )
        st.plotly_chart(chart, use_container_width=True)

        with st.expander("How to use this tab"):
            st.markdown(
                "- **Z-Score window**: shorter windows react faster but miss longer "
                "drifts; longer windows are more stable.\n"
                "- **STL period**: 7 = weekly seasonality (workday/weekend pattern).\n"
                "- **IForest contamination**: prior on the fraction of anomalies; "
                "raise this if recall is low.\n"
                "- Compare the live F1 here to the bar chart in *Detector "
                "comparison* to see how sensitive each detector is to its tuning."
            )

    with tab6:
        st.subheader("🎬 Day-by-day replay")
        st.caption(
            "Walks through the 90-day dataset one day at a time. Anomalies appear "
            "as they would in real-life monitoring — useful for showing reviewers "
            "the 'catch in hours not weeks' story."
        )
        replay_detector = st.selectbox(
            "Detector to replay",
            options=active_detectors,
            format_func=lambda x: DETECTOR_LABELS[x],
            key="replay_det",
        )
        det = detections_by_name[replay_detector]
        all_dates = sorted(pd.to_datetime(daily["date"]).unique())
        if len(all_dates) < 2:
            st.info("Need more than one day of data for replay.")
        else:
            color = DETECTOR_STYLES[replay_detector]["color"]
            y_max = float(daily["cost"].max()) * 1.15
            x_range = [all_dates[0], all_dates[-1]]

            daily_sorted = daily.sort_values("date").reset_index(drop=True)
            det_anom = det[det["is_anomaly"]].copy()
            det_anom["date"] = pd.to_datetime(det_anom["date"])
            anom_dates_sorted = sorted(det_anom["date"].unique())

            frames = []
            for cutoff in all_dates:
                line_view = daily_sorted[daily_sorted["date"] <= cutoff]
                visible_anom_dates = [d for d in anom_dates_sorted if d <= cutoff]
                anom_costs = (
                    daily_sorted[daily_sorted["date"].isin(visible_anom_dates)]
                    .drop_duplicates("date")
                    .sort_values("date")
                )
                frames.append(go.Frame(
                    name=str(pd.Timestamp(cutoff).date()),
                    data=[
                        go.Scatter(
                            x=line_view["date"], y=line_view["cost"],
                            mode="lines+markers",
                            line=dict(color="#94A3B8", width=2),
                            marker=dict(size=4, color="#94A3B8"),
                            name="Daily cost",
                        ),
                        go.Scatter(
                            x=anom_costs["date"], y=anom_costs["cost"],
                            mode="markers",
                            marker=dict(color=color, size=14, symbol="x", line=dict(width=2, color="white")),
                            name=f"Anomalies — {DETECTOR_LABELS[replay_detector]}",
                        ),
                    ],
                ))

            initial = frames[0]
            replay_fig = go.Figure(
                data=initial.data,
                frames=frames,
            )
            slider_steps = [
                {
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f.name,
                    "method": "animate",
                }
                for f in frames
            ]
            replay_fig.update_layout(
                title=f"Replay — {DETECTOR_LABELS[replay_detector]}",
                xaxis=dict(title="Date", range=x_range),
                yaxis=dict(title="Cost ($)", range=[0, y_max]),
                height=480,
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.15),
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                         "label": "▶ Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                         "label": "⏸ Pause", "method": "animate"},
                    ],
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0, "y": 1.18,
                    "xanchor": "left", "yanchor": "top",
                    "pad": {"t": 4, "r": 8},
                }],
                sliders=[{
                    "active": 0,
                    "currentvalue": {"prefix": "Day: "},
                    "steps": slider_steps,
                    "pad": {"t": 30, "b": 10},
                    "x": 0.05, "len": 0.9,
                }],
            )
            st.plotly_chart(replay_fig, use_container_width=True)
            st.caption(
                "Tip: hit ▶ Play, or drag the slider. The grey line grows day by day; "
                "red ✕ markers persist as anomalies are caught — each one would have "
                "paged FinOps in real life."
            )

    with tab7:
        st.subheader("Synthetic CUR rows (sample)")
        st.dataframe(cur_df.head(200), use_container_width=True, hide_index=True)
        st.subheader("Ground-truth labels")
        gt = labels_df[labels_df["is_anomaly"]].copy()
        gt["date"] = gt["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(gt, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
