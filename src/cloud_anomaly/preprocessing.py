"""Preprocessing helpers: load raw CUR rows and aggregate into daily series."""
from __future__ import annotations

from pathlib import Path

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


def aggregate_by_service(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format daily spend per service.

    Returns columns: ``date``, ``service``, ``cost``.
    """
    long = (
        df.groupby(["date", "service"], as_index=False)["cost"]
        .sum()
        .sort_values(["service", "date"])
    )
    filled = []
    for service, sub in long.groupby("service"):
        sub = _fill_missing_days(sub[["date", "cost"]], "cost")
        sub["service"] = service
        filled.append(sub)
    return pd.concat(filled, ignore_index=True)[["date", "service", "cost"]]


def pivot_services(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-format pivot: one column per service, indexed by date."""
    long = aggregate_by_service(df)
    wide = long.pivot(index="date", columns="service", values="cost").fillna(0.0)
    wide.columns.name = None
    return wide


def _fill_missing_days(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reindex to a continuous daily range and forward-fill ``value_col``."""
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    out = df.set_index("date").reindex(full)
    out[value_col] = out[value_col].ffill().fillna(0.0)
    out.index.name = "date"
    return out.reset_index()
