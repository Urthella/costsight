"""Render presentation-ready figures from a fresh pipeline run.

Outputs land in ``slides/figures/`` as 1600x900 PNGs that can be dropped
straight into the existing slide deck:

    fig01_dataset_overview.png   — slide 6 replacement (synthetic data peek)
    fig02_f1_by_type.png         — slide 9 replacement (empirical bar chart)
    fig03_performance_matrix.png — slide 10 replacement (empirical heatmap)
    fig04_detector_overlay.png   — bonus: same trend, three detectors
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cloud_anomaly.detectors import DETECTORS  # noqa: E402
from cloud_anomaly.preprocessing import aggregate_by_service, aggregate_daily  # noqa: E402
from cloud_anomaly.synthetic_data import generate  # noqa: E402

OUT = ROOT / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Project palette — matches the proposal slide deck.
TEAL = "#2A9D8F"
ORANGE = "#E9A23B"
SLATE = "#264653"
RED = "#C0392B"
LIGHT = "#F4F1EB"


def _figsize(w_in=16, h_in=9):
    return (w_in, h_in)


def fig_dataset_overview(cur, labels):
    """Daily total cost, with each anomaly window highlighted."""
    daily = aggregate_daily(cur)
    fig, ax = plt.subplots(figsize=_figsize(), dpi=110)
    ax.plot(daily["date"], daily["cost"], color=TEAL, lw=2.4, label="Total daily cost")

    type_colors = {
        "point_spike": RED,
        "level_shift": ORANGE,
        "gradual_drift": "#9B59B6",
    }
    plotted = set()
    by_type = labels[labels["is_anomaly"]].copy()
    for atype, sub in by_type.groupby("anomaly_type"):
        spans = []
        for _, row in sub.sort_values("date").iterrows():
            if not spans or row["date"] != spans[-1][1] + pd.Timedelta(days=1):
                spans.append([row["date"], row["date"]])
            else:
                spans[-1][1] = row["date"]
        for start, end in spans:
            label = atype.replace("_", " ").title() if atype not in plotted else None
            ax.axvspan(start, end + pd.Timedelta(days=1),
                       color=type_colors[atype], alpha=0.22, label=label)
            plotted.add(atype)

    ax.set_title("Synthetic AWS CUR — daily spend with injected anomalies",
                 fontsize=20, fontweight="bold", color=SLATE, pad=14)
    ax.set_ylabel("$ / day", fontsize=14, color=SLATE)
    ax.set_xlabel("Date", fontsize=14, color=SLATE)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    fig.tight_layout()
    out = OUT / "fig01_dataset_overview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_f1_by_type(comparison):
    """Grouped bar chart: detectors × anomaly type, height = F1."""
    df = comparison[comparison["anomaly_type"] != "OVERALL"].copy()
    types = ["point_spike", "level_shift", "gradual_drift"]
    detectors = ["zscore", "stl", "iforest"]
    label_map = {"zscore": "Z-Score", "stl": "STL Decomposition", "iforest": "Isolation Forest"}
    color_map = {"zscore": SLATE, "stl": TEAL, "iforest": ORANGE}

    fig, ax = plt.subplots(figsize=_figsize(), dpi=110)
    width = 0.25
    x = np.arange(len(types))
    for i, det in enumerate(detectors):
        vals = [
            df[(df["detector"] == det) & (df["anomaly_type"] == t)]["f1"].iloc[0]
            for t in types
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width=width,
                      color=color_map[det], label=label_map[det])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", fontsize=12, fontweight="bold",
                    color=SLATE)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in types], fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 score", fontsize=14, color=SLATE)
    ax.set_title("Empirical F1 by anomaly type — same dataset, three detectors",
                 fontsize=20, fontweight="bold", color=SLATE, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig02_f1_by_type.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_performance_matrix(comparison):
    """Heatmap-style matrix: rows = anomaly types, cols = detectors, cell = F1."""
    types = ["point_spike", "level_shift", "gradual_drift"]
    detectors = ["zscore", "stl", "iforest"]
    label_d = {"zscore": "Z-Score", "stl": "STL", "iforest": "Isolation Forest"}
    label_t = {"point_spike": "Point Spike", "level_shift": "Level Shift",
               "gradual_drift": "Gradual Drift"}

    grid = np.zeros((len(types), len(detectors)))
    for i, t in enumerate(types):
        for j, d in enumerate(detectors):
            row = comparison[(comparison["anomaly_type"] == t) & (comparison["detector"] == d)]
            grid[i, j] = row["f1"].iloc[0]

    fig, ax = plt.subplots(figsize=_figsize(14, 8), dpi=110)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(types)):
        for j in range(len(detectors)):
            ax.text(j, i, f"{grid[i, j]:.2f}",
                    ha="center", va="center", fontsize=22, fontweight="bold",
                    color="black" if 0.25 < grid[i, j] < 0.75 else "white")
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels([label_d[d] for d in detectors], fontsize=14)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([label_t[t] for t in types], fontsize=14)
    ax.set_title("Anomaly type × Method — empirical F1 score",
                 fontsize=20, fontweight="bold", color=SLATE, pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("F1 score", fontsize=12, color=SLATE)
    fig.tight_layout()
    out = OUT / "fig03_performance_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_detector_overlay(cur):
    """Three small-multiples panels — one detector per row — showing where each
    fires against the same daily total cost. Far easier to read than overlay.
    """
    long = aggregate_by_service(cur)
    daily = aggregate_daily(cur)
    detector_cfg = [
        ("Z-Score", "zscore", SLATE),
        ("STL Decomposition", "stl", TEAL),
        ("Isolation Forest", "iforest", ORANGE),
    ]

    fig, axes = plt.subplots(3, 1, figsize=_figsize(16, 10), dpi=110, sharex=True)
    for ax, (label, key, color) in zip(axes, detector_cfg):
        det = DETECTORS[key](long)
        flagged_dates = det.loc[det["is_anomaly"], "date"].unique()
        flagged_daily = daily[daily["date"].isin(flagged_dates)]
        n_flagged = len(flagged_daily)

        ax.plot(daily["date"], daily["cost"], color="#888", lw=1.6, alpha=0.7)
        ax.scatter(flagged_daily["date"], flagged_daily["cost"],
                   color=color, s=130, marker="o", alpha=0.95,
                   edgecolor="white", linewidth=1.6, zorder=5)
        ax.set_title(f"{label}  —  {n_flagged} day(s) flagged",
                     fontsize=15, fontweight="bold", color=SLATE, loc="left")
        ax.set_ylabel("$ / day", fontsize=12, color=SLATE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Date", fontsize=12, color=SLATE)
    fig.suptitle("Where each detector fires on the same data",
                 fontsize=20, fontweight="bold", color=SLATE, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "fig04_detector_overlay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    cur, labels, _ = generate()

    long = aggregate_by_service(cur)
    detector_outputs = {name: fn(long) for name, fn in DETECTORS.items()}
    from cloud_anomaly.evaluation import compare_detectors
    comparison = compare_detectors(detector_outputs, labels)

    paths = [
        fig_dataset_overview(cur, labels),
        fig_f1_by_type(comparison),
        fig_performance_matrix(comparison),
        fig_detector_overlay(cur),
    ]
    print("Wrote:")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
