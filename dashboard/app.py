"""Streamlit dashboard for the cloud cost anomaly detector.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
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
    bootstrap_f1_ci,
    compare_detectors,
    cost_saved_estimate,
    evaluate_alerts,
    evaluate_by_type,
    paired_significance,
    time_to_detect,
)
from cloud_anomaly.forecast import (  # noqa: E402
    forecast_per_service,
    projected_monthly_spend,
)
from cloud_anomaly.notification import (  # noqa: E402
    build_payload_from_alert,
    send_webhook,
)
from cloud_anomaly.playbook import PLAYBOOKS  # noqa: E402
from cloud_anomaly.pricing import (  # noqa: E402
    PRICING_SNAPSHOT_DATE,
    estimated_monthly,
    lookup,
)
from cloud_anomaly.clustering import cluster_alerts, summarize_incidents  # noqa: E402
from cloud_anomaly.perf import benchmark_grid  # noqa: E402
from cloud_anomaly.detectors.zscore import detect as zscore_detect_raw  # noqa: E402
from cloud_anomaly.detectors.stl import detect as stl_detect_raw  # noqa: E402
from cloud_anomaly.detectors.iforest import detect as iforest_detect_raw  # noqa: E402
from cloud_anomaly.preprocessing import aggregate_by_service, aggregate_daily, load_cur  # noqa: E402
from cloud_anomaly.synthetic_data import SCENARIOS, generate  # noqa: E402
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
def _load(regenerate: bool, n_days: int, seed: int, scenario: str = "default"):
    if regenerate or not (RAW_DIR / "cur_synthetic.parquet").exists():
        cur_df, labels_df, _ = generate(n_days=n_days, seed=seed, scenario=scenario)
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
        scenario = st.selectbox(
            "Scenario preset",
            options=list(SCENARIOS.keys()),
            format_func=lambda s: f"{s} — {SCENARIOS[s]}"[:60] + ("…" if len(SCENARIOS[s]) > 32 else ""),
            help="Each preset biases the anomaly mix injected into the synthetic data.",
        )
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

    cur_df, labels_df, long, daily = _load(regenerate, n_days, int(seed), scenario)
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

    tabs = st.tabs([
        "📈 Cost trend", "🚨 Alert log", "🔎 Root-cause",
        "📊 Detector comparison", "📅 Calendar", "📉 Forecast",
        "💰 Budget", "📘 Playbook", "🧩 Incidents", "⚡ Perf",
        "🔬 Lab", "🎬 Replay", "🗂️ Raw data",
    ])
    (
        tab1, tab2, tab3, tab4, tab5, tab6,
        tab_budget, tab_playbook, tab_incidents, tab_perf,
        tab7, tab8, tab9,
    ) = tabs

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
                    if row.get("n_detectors", 0) >= 2:
                        st.info(
                            "🤝 **Multi-detector consensus** — see the *Playbook* tab "
                            "for the consensus-class recipe (page on-call, 30-min ack)."
                        )
                    else:
                        st.info(
                            "📘 See the *Playbook* tab for the matching recipe — "
                            "this looks like a **point spike**; the playbook lists the "
                            "CloudTrail / autoscaler checks to run."
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
        st.subheader("Statistical rigor — bootstrap CIs and pairwise significance")
        st.caption(
            "Loaded from `outputs/benchmark_raw.csv` (the per-seed table from "
            "`scripts/run_benchmark.py`). Bootstrap = 2000 resamples, 95% CI. "
            "Pairwise test = Wilcoxon signed-rank on per-seed F1, two-sided."
        )
        bench_raw_path = ROOT / "outputs" / "benchmark_raw.csv"
        if not bench_raw_path.exists():
            st.info(
                "No `outputs/benchmark_raw.csv` yet — run `python scripts/run_benchmark.py` "
                "to populate the per-seed table."
            )
        else:
            raw_runs = pd.read_csv(bench_raw_path)
            ci_rows = []
            for det in ["zscore", "stl", "iforest"]:
                for atype in ["point_spike", "level_shift", "gradual_drift", "OVERALL"]:
                    ci = bootstrap_f1_ci(raw_runs, det, atype)
                    ci_rows.append({
                        "detector": det, "anomaly_type": atype,
                        "F1 mean": round(ci["mean"], 3),
                        "95% CI lo": round(ci["lo"], 3),
                        "95% CI hi": round(ci["hi"], 3),
                        "n seeds": ci["n"],
                    })
            ci_df = pd.DataFrame(ci_rows)
            st.dataframe(ci_df, use_container_width=True, hide_index=True)

            st.markdown("**Pairwise Wilcoxon (OVERALL F1) — is the difference significant?**")
            sig_rows = []
            pairs = [("stl", "iforest"), ("stl", "zscore"), ("iforest", "zscore")]
            for a, b in pairs:
                res = paired_significance(raw_runs, a, b, anomaly_type="OVERALL")
                sig_rows.append({
                    "pair": f"{DETECTOR_LABELS[a].split(' ')[0]} vs {DETECTOR_LABELS[b].split(' ')[0]}",
                    "median F1 delta": round(res.get("median_delta", float("nan")), 4),
                    "Wilcoxon W": round(res["statistic"], 2) if not pd.isna(res["statistic"]) else "—",
                    "p-value": f"{res['p_value']:.2e}" if not pd.isna(res["p_value"]) else "—",
                    "n": res["n"],
                    "verdict": "significant (p<0.05)" if (
                        not pd.isna(res["p_value"]) and res["p_value"] < 0.05
                    ) else "not significant",
                })
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

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
        st.subheader("📅 Cost calendar heatmap")
        st.caption(
            "One cell per (service, day). Color intensity = daily cost. The white "
            "dots are days flagged by the active detectors (severity-filtered). "
            "Drift, level shifts, and weekend seasonality jump out at a glance."
        )
        calendar_view = long.copy()
        calendar_view["date"] = pd.to_datetime(calendar_view["date"])
        pivot = calendar_view.pivot_table(
            index="service", columns="date", values="cost", aggfunc="sum",
        ).fillna(0.0)
        pivot.columns = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in pivot.columns]
        heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="YlOrRd",
            colorbar=dict(title="$"),
            hovertemplate="Service: %{y}<br>Date: %{x}<br>Cost: $%{z:.2f}<extra></extra>",
        ))
        # Overlay detector flags as small white markers at the right (date, service) cell.
        for name in active_detectors:
            flagged = flagged_dates_per_detector[name]
            if not flagged:
                continue
            xs, ys = [], []
            for d, s in flagged:
                xs.append(pd.Timestamp(d).strftime("%Y-%m-%d"))
                ys.append(s)
            style = DETECTOR_STYLES[name]
            heat.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=DETECTOR_LABELS[name],
                marker=dict(
                    color=style["color"], size=8, symbol=style["symbol"],
                    line=dict(width=1, color="white"),
                ),
            ))
        # Ground-truth overlay as small green diamond markers.
        if not labels_df.empty:
            gt = labels_df[labels_df["is_anomaly"]]
            heat.add_trace(go.Scatter(
                x=pd.to_datetime(gt["date"]).dt.strftime("%Y-%m-%d"),
                y=gt["service"],
                mode="markers", name="Ground truth",
                marker=dict(color="#10B981", size=6, symbol="diamond-open",
                            line=dict(width=2, color="#10B981")),
            ))
        heat.update_layout(
            height=440, xaxis_title="Date", yaxis_title="Service",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=80, r=20, t=20, b=80),
        )
        st.plotly_chart(heat, use_container_width=True)

        st.markdown("**Daily total — calendar layout (week × weekday)**")
        daily_view = daily.copy()
        daily_view["date"] = pd.to_datetime(daily_view["date"])
        daily_view["weekday"] = daily_view["date"].dt.day_name().str[:3]
        daily_view["week"] = daily_view["date"].dt.isocalendar().week.astype(int)
        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        cal_pivot = daily_view.pivot_table(
            index="week", columns="weekday", values="cost", aggfunc="sum",
        ).reindex(columns=weekday_order)
        cal = go.Figure(data=go.Heatmap(
            z=cal_pivot.values,
            x=cal_pivot.columns,
            y=[f"W{int(w)}" for w in cal_pivot.index],
            colorscale="Blues",
            colorbar=dict(title="$/day"),
            hovertemplate="Week: %{y}<br>Day: %{x}<br>Cost: $%{z:.0f}<extra></extra>",
        ))
        cal.update_layout(height=380, xaxis_title="Day of week", yaxis_title="ISO week")
        st.plotly_chart(cal, use_container_width=True)

    with tab6:
        st.subheader("📉 Holt-Winters forecast (next 14 days)")
        st.caption(
            "Per-service additive Holt-Winters with weekly seasonality. The shaded "
            "band is a 90% prediction interval from 200 simulated futures. The "
            "projected monthly bill assumes the forecast pace continues for 30 days."
        )
        with st.spinner("Fitting per-service forecasts…"):
            fcast = forecast_per_service(long, horizon=14)
        if fcast.empty:
            st.info("Not enough history for a forecast yet.")
        else:
            services = sorted(fcast["service"].unique())
            chosen = st.multiselect(
                "Services to plot",
                options=services,
                default=services[: min(3, len(services))],
                key="fcast_svc",
            )
            fcast_fig = go.Figure()
            palette = px.colors.qualitative.Set2
            for i, svc in enumerate(chosen):
                hist = fcast[(fcast["service"] == svc) & (fcast["kind"] == "history")]
                future = fcast[(fcast["service"] == svc) & (fcast["kind"] == "forecast")]
                color = palette[i % len(palette)]
                fcast_fig.add_trace(go.Scatter(
                    x=hist["date"], y=hist["cost"], mode="lines",
                    name=f"{svc} — history", line=dict(color=color, width=2),
                ))
                fcast_fig.add_trace(go.Scatter(
                    x=future["date"], y=future["cost"], mode="lines",
                    name=f"{svc} — forecast",
                    line=dict(color=color, dash="dash", width=2),
                ))
                fcast_fig.add_trace(go.Scatter(
                    x=list(future["date"]) + list(future["date"][::-1]),
                    y=list(future["upper"]) + list(future["lower"][::-1]),
                    fill="toself", fillcolor=color, opacity=0.18,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, hoverinfo="skip",
                    name=f"{svc} — 90% PI",
                ))
            fcast_fig.update_layout(
                xaxis_title="Date", yaxis_title="Cost ($)", height=460,
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fcast_fig, use_container_width=True)

            proj = projected_monthly_spend(fcast)
            if not proj.empty:
                st.markdown("**Projected monthly bill (forecast pace × 30 days)**")
                proj_display = proj.copy()
                proj_display["forecast_total"] = proj_display["forecast_total"].round(2)
                proj_display["daily_avg"] = proj_display["daily_avg"].round(2)
                proj_display["projected_monthly"] = proj_display["projected_monthly"].round(2)
                st.dataframe(proj_display, use_container_width=True, hide_index=True)
                k1, k2 = st.columns(2)
                k1.metric(
                    "Projected total monthly spend",
                    f"${proj['projected_monthly'].sum():,.0f}",
                )
                k2.metric(
                    "Top service by forecast",
                    proj.iloc[0]["service"],
                    f"${proj.iloc[0]['projected_monthly']:,.0f}/mo",
                )

    with tab_budget:
        st.subheader("💰 What-if budget tracker")
        st.caption(
            "Set a monthly cap; the dashboard projects when (and which "
            "service) will trip it given the current Holt-Winters forecast. "
            "Real-pricing sanity column compares the forecast pace against "
            f"the AWS Pricing snapshot from {PRICING_SNAPSHOT_DATE}."
        )
        budget = st.number_input(
            "Monthly budget cap (USD)",
            min_value=100.0, value=10000.0, step=500.0, format="%.0f",
        )
        with st.spinner("Forecasting…"):
            fcast_b = forecast_per_service(long, horizon=14)
        proj = projected_monthly_spend(fcast_b)
        if proj.empty:
            st.info("Not enough history to project monthly spend.")
        else:
            current_observed = (
                cur_df.copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .groupby("service", as_index=False)["cost"].sum()
                .rename(columns={"cost": "actual_to_date"})
            )
            joined = proj.merge(current_observed, on="service", how="left")
            joined["actual_to_date"] = joined["actual_to_date"].fillna(0.0)
            joined["projected_total"] = joined["projected_monthly"]
            joined["pct_of_budget"] = (joined["projected_total"] / budget * 100).round(1)
            joined["over_budget"] = joined["projected_total"] > budget
            joined["realistic_unit"] = joined["service"].apply(
                lambda s: ", ".join(
                    f"{q.sku} (${q.unit_price})" for q in lookup(s)[:1]
                )
            )

            st.dataframe(
                joined[[
                    "service", "actual_to_date", "projected_monthly",
                    "pct_of_budget", "over_budget", "realistic_unit",
                ]].round(2),
                use_container_width=True, hide_index=True,
            )

            total_proj = float(joined["projected_total"].sum())
            burn_pct = total_proj / budget * 100 if budget else 0
            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("Total projected monthly spend", f"${total_proj:,.0f}")
            kc2.metric("% of budget", f"{burn_pct:.1f}%",
                       delta=f"${total_proj - budget:+,.0f} vs cap")
            kc3.metric("Services over budget", int(joined["over_budget"].sum()))

            burn_chart = px.bar(
                joined.sort_values("projected_total", ascending=False),
                x="service", y="projected_total",
                color="over_budget",
                color_discrete_map={True: "#EF4444", False: "#10B981"},
                title="Projected monthly spend by service",
                height=380,
            )
            burn_chart.add_hline(
                y=budget, line_dash="dash", line_color="#F59E0B",
                annotation_text=f"Budget cap: ${budget:,.0f}",
                annotation_position="top right",
            )
            st.plotly_chart(burn_chart, use_container_width=True)

            with st.expander("Days-to-cap forecast (linear extrapolation)"):
                days_elapsed = max(int(cur_df["date"].nunique()), 1)
                burn_per_day = float(current_observed["actual_to_date"].sum()) / days_elapsed
                if burn_per_day > 0:
                    days_remaining = max(0.0, (budget - float(current_observed["actual_to_date"].sum())) / burn_per_day)
                    st.write(
                        f"At the current burn rate (**${burn_per_day:,.0f}/day**), "
                        f"the remaining budget lasts **{days_remaining:.1f} days**."
                    )
                else:
                    st.write("Burn rate too low to estimate runway.")

    with tab_playbook:
        st.subheader("📘 Anomaly playbook")
        st.caption(
            "When a detector fires, *what should the FinOps engineer do?* "
            "Each playbook entry is a deterministic recipe keyed off the "
            "anomaly type — owner, SLA, and a numbered checklist."
        )
        for atype, book in PLAYBOOKS.items():
            with st.expander(f"**{atype}** — {book['headline']}", expanded=(atype == "point_spike")):
                st.markdown(f"**Owner:** {book['owner']}")
                st.markdown(f"**SLA:** {book['sla']}")
                st.markdown("**Checks:**")
                st.markdown(book["checks"])

        st.markdown("---")
        st.subheader("📨 Send a sample alert (webhook test)")
        st.caption(
            "Posts the highest-severity alert as a Slack-shaped JSON payload "
            "to the URL you provide. Use a private webhook (or "
            "https://webhook.site/) — the dashboard does NOT keep the URL."
        )
        if filtered_alerts.empty:
            st.info("No alerts to send right now.")
        else:
            top = filtered_alerts.sort_values("severity_score", ascending=False).iloc[0]
            payload = build_payload_from_alert(
                top, detector=str(top.get("detector", "stl")),
                runbook=PLAYBOOKS.get(
                    "point_spike" if top.get("severity") == "HIGH" else "level_shift",
                    PLAYBOOKS["point_spike"],
                )["headline"],
            )
            st.code(json.dumps(payload.to_slack_block(), indent=2), language="json")
            url = st.text_input("Webhook URL (optional — leave blank to dry-run)", value="")
            if st.button("Send alert"):
                if not url:
                    st.warning("Dry-run: payload above is what would be sent.")
                else:
                    with st.spinner("POSTing…"):
                        result = send_webhook(payload, url)
                    if result["status"] == "ok":
                        st.success(f"Sent · HTTP {result.get('code')}")
                    else:
                        st.error(f"Failed · {result}")

    with tab_incidents:
        st.subheader("🧩 Alert clustering — turn rows into incidents")
        st.caption(
            "DBSCAN over (day, service, severity, detector) groups close-in-time "
            "alerts into incidents. The default DBSCAN density radius (eps) is "
            "tuned for 90-day datasets with ~3-5 anomaly windows; tweak it if "
            "you change the synthetic scenario."
        )
        if filtered_alerts.empty:
            st.info("No alerts to cluster.")
        else:
            ec1, ec2 = st.columns(2)
            eps = ec1.slider("DBSCAN eps", 0.3, 2.0, 0.85, 0.05, key="dbs_eps")
            min_samples = ec2.slider("Min samples", 2, 5, 2, 1, key="dbs_min")

            clustered = cluster_alerts(filtered_alerts, eps=eps, min_samples=min_samples)
            incidents = summarize_incidents(clustered)
            n_incidents = int(incidents["incident_id"].nunique()) if not incidents.empty else 0
            n_singletons = int((clustered["incident_id"] == -1).sum())
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Alerts", len(clustered))
            ic2.metric("Incidents", n_incidents)
            ic3.metric("Singletons (un-clustered)", n_singletons)

            if incidents.empty:
                st.info("No multi-alert incidents at this eps. Try raising eps.")
            else:
                view = incidents.copy()
                view["first_date"] = view["first_date"].dt.strftime("%Y-%m-%d")
                view["last_date"] = view["last_date"].dt.strftime("%Y-%m-%d")
                st.dataframe(view, use_container_width=True, hide_index=True)

                inc_chart = px.scatter(
                    clustered.assign(date=lambda d: pd.to_datetime(d["date"])),
                    x="date", y="service", color="incident_id",
                    size="severity_score", hover_data=["severity", "cost"],
                    height=380,
                    title="Alerts colored by incident_id (-1 = singleton)",
                )
                st.plotly_chart(inc_chart, use_container_width=True)

    with tab_perf:
        st.subheader("⚡ Detector performance — measured runtime")
        st.caption(
            "Median of 3 runs per (detector, dataset_size) combination. Used "
            "for production-deploy sizing — see REPORT § 4.1 *Cloud architecture* "
            "for how this maps to ECS Fargate vCPU choices."
        )
        if st.button("Run performance grid (4 sizes × 4 detectors)"):
            with st.spinner("Timing detectors…"):
                perf_df = benchmark_grid()
            st.session_state["perf_df"] = perf_df

        perf_df = st.session_state.get("perf_df")
        if perf_df is None:
            st.info("Click *Run performance grid* to populate this tab.")
        else:
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            chart = px.line(
                perf_df, x="n_days", y="seconds_per_run",
                color="detector", markers=True, height=380,
                title="Seconds per detector run vs dataset size",
                labels={"n_days": "Dataset size (days)", "seconds_per_run": "Seconds"},
            )
            st.plotly_chart(chart, use_container_width=True)

            throughput = px.bar(
                perf_df, x="n_days", y="rows_per_second", color="detector",
                barmode="group", height=320,
                title="Throughput (rows/sec) by detector and dataset size",
            )
            st.plotly_chart(throughput, use_container_width=True)

    with tab7:
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

    with tab8:
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

    with tab9:
        st.subheader("Synthetic CUR rows (sample)")
        st.dataframe(cur_df.head(200), use_container_width=True, hide_index=True)
        st.subheader("Ground-truth labels")
        gt = labels_df[labels_df["is_anomaly"]].copy()
        gt["date"] = gt["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(gt, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
