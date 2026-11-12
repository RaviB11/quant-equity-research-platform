"""Earnings-quality history for one company: accruals, cash conversion and free cash flow trend.

Everything is point in time: for a given month, each fiscal year uses the latest version of
its statement that had been published by then. See docs/earnings_quality.md for what each
indicator means and where it misleads.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .db import table_df

DEFAULT_FLAGS = {"min_cash_conversion": 0.8, "high_accruals_percentile": 0.8, "fcf_decline_years": 3}


def history(conn: sqlite3.Connection, cfg: dict, ticker: str, as_of: pd.Timestamp, years: int = 6) -> pd.DataFrame:
    f = table_df(conn, "SELECT * FROM fundamentals WHERE ticker = ? AND source = ? AND available_date <= ? "
                       "ORDER BY period_end, available_date",
                 (ticker.upper(), cfg["fundamentals"]["source"], pd.Timestamp(as_of).strftime("%Y-%m-%d")),
                 date_cols=("period_end", "available_date"))
    if f.empty:
        return f
    f = f.groupby("period_end").tail(1).sort_values("period_end").reset_index(drop=True)
    gap = f["period_end"].diff().dt.days.between(300, 430)
    pos = lambda s: s.where(s > 0)  # noqa: E731
    f["fcf"] = f["operating_cash_flow"] - f["capex"]
    f["fcf_margin"] = f["fcf"] / pos(f["revenue"])
    f["cash_conversion"] = f["operating_cash_flow"] / pos(f["net_income"])
    avg_assets = ((f["total_assets"] + f["total_assets"].shift(1)) / 2).where(gap)
    f["accruals_ratio"] = (f["net_income"] - f["operating_cash_flow"]) / pos(avg_assets)
    prev = f["fcf"].shift(1)
    f["fcf_growth"] = (f["fcf"] / prev - 1).where(gap & (prev > 0) & (f["fcf"] > 0))
    cols = ["period_end", "available_date", "version_type", "revenue", "net_income", "operating_cash_flow", "capex",
            "fcf", "fcf_margin", "fcf_growth", "cash_conversion", "accruals_ratio"]
    return f[cols].tail(years).reset_index(drop=True)


def summarize(hist: pd.DataFrame, cross_section: pd.DataFrame | None, ticker: str, cfg: dict) -> dict:
    """Trend statistics and mechanical flags. `cross_section` = all stocks that month (for percentiles)."""
    th = {**DEFAULT_FLAGS, **cfg.get("research", {}).get("flags", {})}
    out: dict = {"years": len(hist), "flags": []}
    if hist.empty:
        return out
    fcf = hist["fcf"].dropna()
    out["fcf_positive_years"] = int((fcf > 0).sum())
    out["fcf_years"] = int(len(fcf))
    valid = hist.dropna(subset=["fcf"])
    if len(valid) >= 2 and valid["fcf"].iloc[0] > 0 and valid["fcf"].iloc[-1] > 0:
        yrs = (valid["period_end"].iloc[-1] - valid["period_end"].iloc[0]).days / 365.25
        out["fcf_cagr"] = (valid["fcf"].iloc[-1] / valid["fcf"].iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    cc = hist["cash_conversion"].dropna()
    out["median_cash_conversion"] = float(cc.median()) if len(cc) else np.nan
    out["latest_accruals"] = hist["accruals_ratio"].iloc[-1]
    if cross_section is not None and "accruals_ratio" in cross_section and pd.notna(out["latest_accruals"]):
        acc = cross_section["accruals_ratio"].dropna()
        mine = cross_section.loc[cross_section["ticker"] == ticker, "accruals_ratio"]
        if len(acc) >= 20 and len(mine) and pd.notna(mine.iloc[0]):
            out["accruals_percentile"] = float((acc <= mine.iloc[0]).mean())

    last3 = hist["cash_conversion"].tail(3)
    weak = int((last3 < th["min_cash_conversion"]).sum())
    if weak >= 2:
        out["flags"].append(f"Operating cash flow below {th['min_cash_conversion']:.0%} of net income in {weak} of the "
                            "last 3 years: profits are running ahead of cash.")
    if out.get("accruals_percentile", 0) >= th["high_accruals_percentile"]:
        out["flags"].append(f"Accruals ratio in the highest {1 - th['high_accruals_percentile']:.0%} of the universe "
                            "this month: an unusually large share of profit is not yet cash.")
    n = int(th["fcf_decline_years"])
    tail = fcf.tail(n + 1)
    if len(tail) == n + 1 and (tail.diff().dropna() < 0).all():
        out["flags"].append(f"Free cash flow fell in each of the last {n} years.")
    if len(fcf) and fcf.iloc[-1] < 0:
        out["flags"].append("Free cash flow was negative in the latest fiscal year.")
    return out
