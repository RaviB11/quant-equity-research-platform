"""Valuation ratios, factor exposures and cross-sectional scores.

Pipeline: SQL point-in-time join -> split adjustment -> ratios -> momentum and
forward returns -> winsorised, (sector-neutral) z-scores -> group scores ->
composite and labels. Forward returns are attached for evaluation only; no
scoring function reads them.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .db import read_sql_file, table_df

# raw factor column -> factor group. Higher raw value = more attractive for all of these.
RAW_FACTORS = {
    "earnings_yield": "value",
    "ebit_ev": "value",
    "ebitda_ev": "value",
    "book_to_market": "value",
    "momentum_12_1": "momentum",
    "roe": "quality",
    "accruals_low": "earnings_quality",
}
GROUPS = ("value", "momentum", "quality", "earnings_quality")


# --------------------------------------------------------------------------- inputs
def split_factor(df: pd.DataFrame, splits: pd.DataFrame, date_col: str,
                 ticker_col: str = "ticker") -> pd.Series:
    """Product of split ratios dated strictly after ``df[date_col]``, per row.

    Yahoo's `Close` is restated for every split up to today, while filings report
    shares/EPS in the units of their filing date. Multiplying as-reported shares
    (and dividing EPS) by this factor puts both on the same basis.
    """
    out = pd.Series(1.0, index=df.index)
    if splits is None or splits.empty:
        return out
    dates = pd.to_datetime(df[date_col])
    for tk, sp in splits.groupby("ticker"):
        mask = (df[ticker_col] == tk) & dates.notna()
        if not mask.any():
            continue
        sp = sp.sort_values("split_date")
        sdates = pd.to_datetime(sp["split_date"]).to_numpy(dtype="datetime64[ns]")
        ratios = sp["ratio"].to_numpy(dtype=float)
        suffix = np.append(np.cumprod(ratios[::-1])[::-1], 1.0)   # suffix[k] = prod(ratios[k:])
        idx = np.searchsorted(sdates, dates[mask].to_numpy(dtype="datetime64[ns]"), side="right")
        out.loc[mask] = suffix[idx]
    return out


class LookaheadError(RuntimeError):
    """A statement would be used before it was public. Raised explicitly (not via assert),
    so the guarantee also holds under `python -O`."""


def check_point_in_time(df: pd.DataFrame) -> None:
    for col, what in (("available_date", "fundamentals"), ("available_date_prev", "prior-year figures")):
        if col in df and (df[col] > df["month_end"]).any():
            n = int((df[col] > df["month_end"]).sum())
            raise LookaheadError(f"lookahead: {what} used before they were available in {n} stock-months")


def load_factor_inputs(conn: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    params = {"source": cfg["fundamentals"]["source"],
              "max_staleness_days": int(cfg["fundamentals"]["max_staleness_months"]) * 31}
    df = table_df(conn, read_sql_file("factor_inputs.sql"), params,
                  date_cols=("month_end", "period_end", "available_date", "period_end_prev", "available_date_prev"))
    # Point-in-time guarantee: nothing used at month t was published after t.
    check_point_in_time(df)
    return df


LABELS = {"top": "Research candidate", "middle": "Neutral", "bottom": "Lower priority", "none": "Not scored"}


def split_adjust_enabled(cfg: dict) -> bool:
    """`fundamentals.split_adjust` may be one bool or a per-source mapping."""
    setting = cfg["fundamentals"].get("split_adjust", True)
    if isinstance(setting, dict):
        return bool(setting.get(cfg["fundamentals"]["source"], True))
    return bool(setting)


def price_panel(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wide month-end adj_close (rows = full calendar of month-ends, cols = tickers incl. benchmark)."""
    px = table_df(conn, "SELECT ticker, month_end, adj_close FROM prices_monthly", date_cols=("month_end",))
    wide = px.pivot(index="month_end", columns="ticker", values="adj_close").sort_index()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="ME")
    return wide.reindex(full)


def momentum_and_forward(adj_wide: pd.DataFrame, horizons=(1, 6, 12)) -> pd.DataFrame:
    """12-1 momentum (skip the latest month to avoid short-term reversal) and forward returns."""
    out = {"momentum_12_1": adj_wide.shift(1) / adj_wide.shift(12) - 1}
    for h in horizons:
        out[f"ret_fwd_{h}m"] = adj_wide.shift(-h) / adj_wide - 1
    frames = []
    for name, wide in out.items():
        s = wide.stack()
        s.index.names = ["month_end", "ticker"]
        frames.append(s.rename(name))
    return pd.concat(frames, axis=1).reset_index()


# --------------------------------------------------------------------------- ratios
def compute_ratios(df: pd.DataFrame, splits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    if split_adjust_enabled(cfg):
        f_now = split_factor(df, splits, "available_date")
        f_prev = split_factor(df, splits, "available_date_prev")
    else:
        f_now = f_prev = pd.Series(1.0, index=df.index)

    shares = df["shares_diluted"] * f_now
    eps = df["eps_diluted"] / f_now
    # Fill gaps from the accounting identity EPS = NI / shares.
    eps = eps.where(eps.notna(), df["net_income"] / shares)
    shares = shares.where(shares.notna(), (df["net_income"] / eps).where(eps.abs() > 1e-9))
    eps_prev_shares = df["shares_prev"] * f_prev
    eps_prev = (df["eps_prev"] / f_prev)
    eps_prev = eps_prev.where(eps_prev.notna(), df["net_income_prev"] / eps_prev_shares)

    pos = lambda s: s.where(s > 0)  # noqa: E731
    df["shares_adj"] = shares
    df["eps_adj"] = eps
    df["market_cap"] = df["close"] * shares
    # A missing debt or cash fact is not a zero. By default EV (and every EV-based
    # multiple and factor) is left blank; `assume_zero` restores the old shortcut.
    debt, cash = df["total_debt"], df["cash"]
    df["ev_inputs_complete"] = (debt.notna() & cash.notna()).astype(int)
    if cfg["factors"].get("ev_missing_policy", "exclude") == "assume_zero":
        debt, cash = debt.fillna(0.0), cash.fillna(0.0)
    df["ev"] = df["market_cap"] + debt - cash

    df["pe"] = df["close"] / pos(eps)
    df["pb"] = df["market_cap"] / pos(df["total_equity"])
    df["ev_ebitda"] = pos(df["ev"]) / pos(df["ebitda"])
    df["ev_ebit"] = pos(df["ev"]) / pos(df["ebit"])

    gap_days = (df["period_end"] - df["period_end_prev"]).dt.days
    annual_gap = gap_days.between(300, 430)
    df["eps_growth"] = (eps / pos(eps_prev) - 1).where(annual_gap)
    df["peg"] = df["pe"] / pos(df["eps_growth"] * 100)

    # Yields are the factor inputs: they stay continuous through zero/negative earnings.
    df["earnings_yield"] = eps / df["close"]
    df["ebit_ev"] = df["ebit"] / pos(df["ev"])
    df["ebitda_ev"] = df["ebitda"] / pos(df["ev"])
    df["book_to_market"] = df["total_equity"] / pos(df["market_cap"])
    df["roe"] = df["net_income"] / pos(df["total_equity"])

    # Earnings quality: how much of reported profit is backed by operating cash.
    # Accruals ratio (Sloan 1996 style) = (net income - operating cash flow) / average total assets.
    avg_assets = ((df["total_assets"] + df["total_assets_prev"]) / 2).where(annual_gap)
    df["fcf"] = df["operating_cash_flow"] - df["capex"]
    df["accruals_ratio"] = (df["net_income"] - df["operating_cash_flow"]) / pos(avg_assets)
    df["accruals_low"] = -df["accruals_ratio"]                    # higher = more cash-backed earnings
    df["cash_conversion"] = df["operating_cash_flow"] / pos(df["net_income"])
    df["fcf_yield"] = df["fcf"] / pos(df["market_cap"])
    df["fcf_margin"] = df["fcf"] / pos(df["revenue"])

    excluded = df["sector"].isin(cfg["factors"].get("ev_excluded_sectors", []))
    for col in ("ev", "ev_ebitda", "ev_ebit", "ebit_ev", "ebitda_ev"):
        df.loc[excluded, col] = np.nan
    return df.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- scores
def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.notna().sum() < 5:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if s.notna().sum() < 3 or not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd


def score_cross_section(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc, sc = cfg["factors"], cfg["screen"]
    lo, hi = fc["winsorize"]
    df = df.copy()
    by_date = df["month_end"]

    for col in RAW_FACTORS:
        w = df.groupby("month_end")[col].transform(lambda s: _winsorize(s, lo, hi))
        z = w.groupby(by_date).transform(_zscore)
        if fc.get("sector_neutral", True):
            keys = [by_date, df["sector"]]
            sec_mean = z.groupby(keys).transform("mean")
            sec_n = z.groupby(keys).transform("count")
            z = (z - sec_mean.where(sec_n >= 3, 0.0)).groupby(by_date).transform(_zscore)
        df[f"z_{col}"] = z

    value_cols = [f"z_{c}" for c, g in RAW_FACTORS.items() if g == "value"]
    value_mean = df[value_cols].mean(axis=1).where(df[value_cols].notna().sum(axis=1) >= fc["min_value_subfactors"])
    df["value_score"] = value_mean.groupby(by_date).transform(_zscore)
    df["momentum_score"] = df["z_momentum_12_1"]
    df["quality_score"] = df["z_roe"]
    df["earnings_quality_score"] = df["z_accruals_low"]

    weights = {g: float(fc["weights"].get(g, 0.0)) for g in GROUPS}
    active = [g for g in GROUPS if weights[g] > 0]
    if not active:
        raise ValueError("factors.weights needs at least one positive weight")
    total = sum(weights[g] for g in active)
    comp = sum(df[f"{g}_score"] * (weights[g] / total) for g in active)
    comp = comp.where(df[[f"{g}_score" for g in active]].notna().all(axis=1))
    n_scored = comp.groupby(by_date).transform("count")
    df["composite"] = comp.where(n_scored >= fc["min_names_per_date"])
    df["composite_pct"] = df.groupby("month_end")["composite"].rank(pct=True)
    # Mechanical position in this month's score distribution, not a recommendation.
    df["label"] = np.select(
        [df["composite"].isna(), df["composite_pct"] > sc["candidate_quantile"],
         df["composite_pct"] <= sc["lower_priority_quantile"]],
        [LABELS["none"], LABELS["top"], LABELS["bottom"]], default=LABELS["middle"])

    # Baseline rule-based screen: cheapest on EV/EBIT among positive-momentum names.
    rb = sc["rule_based"]
    eligible = ~df["sector"].isin(rb.get("exclude_sectors", [])) & (df["ebit_ev"] > 0)
    if rb.get("require_positive_momentum", True):
        eligible &= df["momentum_12_1"] > 0
    df["rule_score"] = df["ebit_ev"].where(eligible)
    return df


def membership_flags(conn: sqlite3.Connection, keys: pd.DataFrame) -> pd.Series | None:
    """1 if the stock was an index member at that month-end, 0 if not; None without membership data."""
    spans = table_df(conn, "SELECT * FROM membership", date_cols=("start_date", "end_date"))
    if spans.empty:
        return None
    k = keys[["ticker", "month_end"]].reset_index()
    m = k.merge(spans, on="ticker", how="left")
    # A ticker with no spans merges to one all-NaN row, which counts as a member.
    inside = ((m["start_date"].isna() | (m["start_date"] <= m["month_end"]))
              & (m["end_date"].isna() | (m["month_end"] <= m["end_date"])))
    flag = inside.groupby(m["index"]).any()
    return flag.reindex(keys.index).astype(int)


def build_factor_panel(conn: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    inputs = load_factor_inputs(conn, cfg)
    splits = table_df(conn, "SELECT * FROM splits", date_cols=("split_date",))
    ratios = compute_ratios(inputs, splits, cfg)
    horizons = tuple(sorted(set(cfg["backtest"]["forward_horizons"]) | {1, cfg["model"]["label_horizon_months"]}))
    mom = momentum_and_forward(price_panel(conn), horizons)
    panel = ratios.merge(mom, on=["month_end", "ticker"], how="left")

    # Point-in-time index membership: a stock is only scored in months it was in the index,
    # so the screen cannot pick a company before it joined (e.g. after a strong run-up).
    flags = membership_flags(conn, panel) if cfg["universe"].get("point_in_time_membership", True) else None
    panel["is_member"] = 1 if flags is None else flags
    if flags is not None:
        panel.loc[panel["is_member"] == 0, list(RAW_FACTORS)] = np.nan
    scored = score_cross_section(panel, cfg)
    scored.loc[scored["is_member"] == 0, "rule_score"] = np.nan
    return scored


PANEL_COLUMNS = [
    "month_end", "ticker", "name", "sector", "close", "adj_close", "period_end", "available_date",
    "available_date_is_estimated", "version_type", "accession", "ev_inputs_complete", "is_member", "market_cap", "ev", "pe", "pb", "ev_ebitda", "ev_ebit", "peg",
    "eps_growth", "earnings_yield", "ebit_ev", "ebitda_ev", "book_to_market", "roe", "momentum_12_1",
    "fcf", "fcf_yield", "fcf_margin", "accruals_ratio", "cash_conversion",
    *[f"z_{c}" for c in RAW_FACTORS], "value_score", "momentum_score", "quality_score", "earnings_quality_score",
    "composite", "composite_pct", "label", "rule_score",
]


def ev_coverage(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per month: how many stocks with fundamentals could get EV-based factors, and why not."""
    excluded = panel["sector"].isin(cfg["factors"].get("ev_excluded_sectors", []))
    base = panel[panel["period_end"].notna() & ~excluded]
    g = base.groupby("month_end")
    out = pd.DataFrame({
        "with_fundamentals": g.size(),
        "ev_inputs_complete": g["ev_inputs_complete"].sum(),
        "missing_debt": g["total_debt"].apply(lambda s: int(s.isna().sum())),
        "missing_cash": g["cash"].apply(lambda s: int(s.isna().sum())),
        "ev_factor_available": g["ebit_ev"].count(),
    })
    out["ev_input_coverage"] = out["ev_inputs_complete"] / out["with_fundamentals"]
    out["sector_excluded"] = panel[panel["period_end"].notna() & excluded].groupby("month_end").size()
    return out.fillna({"sector_excluded": 0}).reset_index()


def save_panel(conn: sqlite3.Connection, panel: pd.DataFrame) -> None:
    cols = PANEL_COLUMNS + sorted(c for c in panel.columns if c.startswith("ret_fwd_"))
    out = panel[cols].copy()
    for c in ("month_end", "period_end", "available_date"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    out.to_sql("factor_panel", conn, if_exists="replace", index=False)
    conn.commit()


def load_panel(conn: sqlite3.Connection) -> pd.DataFrame:
    return table_df(conn, "SELECT * FROM factor_panel",
                    date_cols=("month_end", "period_end", "available_date"))
