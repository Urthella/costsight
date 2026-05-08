"""Short-horizon cost forecasting using Holt-Winters exponential smoothing.

Used by the dashboard to answer "if the trend continues, what does the
next 14 days of spend look like, and which services are tracking to
exceed their baseline?". Deliberately simple — same library family as
the STL detector — so the project keeps a coherent modeling story.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_per_service(
    long_df: pd.DataFrame,
    horizon: int = 14,
    seasonal_periods: int = 7,
    n_simulations: int = 200,
    confidence: float = 0.90,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Forecast each service's daily cost for `horizon` days ahead.

    Returns a long-format frame with columns ``date, service, kind, cost,
    lower, upper``. ``kind`` is "history" for observed data and "forecast"
    for the predicted window; ``lower`` / ``upper`` are populated only for
    forecast rows.
    """
    long_df = long_df.copy()
    long_df["date"] = pd.to_datetime(long_df["date"])
    rng = np.random.default_rng(rng_seed)
    alpha = (1 - confidence) / 2

    out_rows: list[dict] = []
    for service, group in long_df.groupby("service"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < 2 * seasonal_periods:
            for _, row in group.iterrows():
                out_rows.append({
                    "date": row["date"], "service": service, "kind": "history",
                    "cost": float(row["cost"]), "lower": np.nan, "upper": np.nan,
                })
            continue

        history = group["cost"].astype(float).to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    history,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True, use_brute=False)
        except Exception:
            for _, row in group.iterrows():
                out_rows.append({
                    "date": row["date"], "service": service, "kind": "history",
                    "cost": float(row["cost"]), "lower": np.nan, "upper": np.nan,
                })
            continue

        point = np.asarray(model.forecast(horizon))
        residuals = history - model.fittedvalues
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        sims = point[None, :] + rng.normal(0, sigma, size=(n_simulations, horizon))
        sims = np.clip(sims, a_min=0.0, a_max=None)
        lower = np.quantile(sims, alpha, axis=0)
        upper = np.quantile(sims, 1 - alpha, axis=0)

        last_date = group["date"].iloc[-1]
        for _, row in group.iterrows():
            out_rows.append({
                "date": row["date"], "service": service, "kind": "history",
                "cost": float(row["cost"]), "lower": np.nan, "upper": np.nan,
            })
        for h in range(horizon):
            f_date = last_date + pd.Timedelta(days=h + 1)
            out_rows.append({
                "date": f_date, "service": service, "kind": "forecast",
                "cost": float(max(point[h], 0.0)),
                "lower": float(max(lower[h], 0.0)),
                "upper": float(max(upper[h], 0.0)),
            })

    return pd.DataFrame(out_rows)


def projected_monthly_spend(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Sum the forecast horizon per service and project to a 30-day month.

    Returns a per-service frame with ``forecast_horizon_days``,
    ``forecast_total``, ``daily_avg``, and ``projected_monthly``.
    """
    fcast = forecast_df[forecast_df["kind"] == "forecast"]
    if fcast.empty:
        return pd.DataFrame(
            columns=[
                "service", "forecast_horizon_days", "forecast_total",
                "daily_avg", "projected_monthly",
            ]
        )
    rows = []
    for service, group in fcast.groupby("service"):
        n = len(group)
        total = float(group["cost"].sum())
        daily = total / n if n else 0.0
        rows.append({
            "service": service,
            "forecast_horizon_days": n,
            "forecast_total": total,
            "daily_avg": daily,
            "projected_monthly": daily * 30,
        })
    return pd.DataFrame(rows).sort_values("projected_monthly", ascending=False).reset_index(drop=True)
