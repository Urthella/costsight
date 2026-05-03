"""Preprocessing helpers: load raw CUR rows and aggregate into daily series."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import RAW_DIR


def load_cur(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic CUR file (Parquet preferred, falls back to CSV)."""
    path = path or RAW_DIR / "cur_synthetic.parquet"
    if path.suffix == ".parquet" and path.exists():
        df = pd.read_parquet(path)
    else:
        csv_path = path.with_suffix(".csv")
        df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Total spend per day across all services.

    Returns columns: ``date``, ``cost``.
    """
    daily = df.groupby("date", as_index=False)["cost"].sum().sort_values("date")
    daily = _fill_missing_days(daily, "cost")
    return daily.reset_index(drop=True)


def aggregate_by(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Long-format daily spend grouped by ``keys`` (e.g. ``['service']`` or
    ``['service', 'env']``).

    Returns columns: ``date``, *keys, ``cost``. Each group is reindexed to
    a continuous daily range and forward-filled so detectors see no gaps.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("aggregate_by requires at least one grouping key")
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"CUR frame is missing grouping keys: {missing}")

    long = (
        df.groupby(["date", *keys], as_index=False)["cost"]
        .sum()
        .sort_values([*keys, "date"])
    )
    filled = []
    for group_vals, sub in long.groupby(keys, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        sub = _fill_missing_days(sub[["date", "cost"]], "cost")
        for k, v in zip(keys, group_vals):
            sub[k] = v
        filled.append(sub)
    return pd.concat(filled, ignore_index=True)[["date", *keys, "cost"]]


def aggregate_by_service(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format daily spend per service (legacy single-key wrapper)."""
    return aggregate_by(df, ["service"])


def pivot_services(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-format pivot: one column per service, indexed by date."""
    long = aggregate_by_service(df)
    wide = long.pivot(index="date", columns="service", values="cost").fillna(0.0)
    wide.columns.name = None
    return wide


def _fill_missing_days(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reindex to a continuous daily range and forward-fill ``value_col``."""
    if df.empty:
        return df
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out = df.set_index("date").reindex(full)
    out[value_col] = out[value_col].ffill().fillna(0.0)
    out.index.name = "date"
    return out.reset_index()
