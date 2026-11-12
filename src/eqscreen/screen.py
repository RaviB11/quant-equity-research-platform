"""Current-period screens written to CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DISPLAY_COLS = [
    "rank", "ticker", "name", "sector", "label", "composite", "composite_pct", "value_score",
    "momentum_score", "quality_score", "close", "market_cap_bn", "pe", "pb", "ev_ebitda", "ev_ebit",
    "peg", "earnings_yield", "ebit_ev", "momentum_12_1", "roe", "fcf_yield", "accruals_ratio",
    "cash_conversion", "fundamentals_period_end",
    "fundamentals_available", "available_date_is_estimated",
]


def latest_scored_date(panel: pd.DataFrame, col: str = "composite") -> pd.Timestamp:
    dates = panel.loc[panel[col].notna(), "month_end"]
    if dates.empty:
        raise ValueError(f"No month has enough names with a {col}; check data coverage and factors.min_names_per_date")
    return dates.max()


def _decorate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_cap_bn"] = df["market_cap"] / 1e9
    df["fundamentals_period_end"] = df["period_end"].dt.strftime("%Y-%m-%d")
    df["fundamentals_available"] = df["available_date"].dt.strftime("%Y-%m-%d")
    return df


def composite_screen(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    as_of = as_of or latest_scored_date(panel)
    cur = panel[(panel["month_end"] == as_of) & panel["composite"].notna()]
    cur = _decorate(cur.sort_values("composite", ascending=False))
    cur["rank"] = range(1, len(cur) + 1)
    return cur[DISPLAY_COLS].reset_index(drop=True)


def rule_based_screen(panel: pd.DataFrame, top_n: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Lowest EV/EBIT among non-excluded sectors with positive 12-1 momentum."""
    as_of = as_of or latest_scored_date(panel, "rule_score")
    cur = panel[(panel["month_end"] == as_of) & panel["rule_score"].notna()]
    cur = _decorate(cur.sort_values("ev_ebit", ascending=True).head(top_n))
    cur["rank"] = range(1, len(cur) + 1)
    return cur[DISPLAY_COLS].reset_index(drop=True)


def write_screens(cfg: dict, panel: pd.DataFrame) -> dict:
    out = Path(cfg["project"]["output_dir"])
    comp = composite_screen(panel)
    rule = rule_based_screen(panel, int(cfg["screen"]["rule_based"]["top_n"]))
    as_of = latest_scored_date(panel)
    comp.to_csv(out / "screen_composite_latest.csv", index=False, float_format="%.6g")
    rule.to_csv(out / "screen_rule_based_latest.csv", index=False, float_format="%.6g")
    comp.to_csv(out / f"screen_composite_{as_of:%Y%m}.csv", index=False, float_format="%.6g")
    return {"as_of": as_of.strftime("%Y-%m-%d"), "n_scored": len(comp),
            "n_candidates": int((comp["label"] == "Research candidate").sum()), "n_rule": len(rule)}
