"""Backtests of factor screens.

Portfolios are formed from the score known at month-end t and held over
(t + lag, t + lag + H]. With lag = 0 the trade is assumed to happen at the same
month-end close the signal was computed from (the usual academic convention, and
optimistic); lag = 1 waits a full month before trading, a conservative bound.
Equal weight at formation, weights drift with returns inside the holding period,
transaction costs are charged on traded weight at rebalance.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- returns
def forward_return_matrix(adj_wide: pd.DataFrame) -> pd.DataFrame:
    """Row t holds each ticker's return over (t, t+1]."""
    return adj_wide.shift(-1) / adj_wide - 1


def select_top(scores: pd.Series, top_q: float, min_holdings: int) -> list[str]:
    s = scores.dropna()
    if s.empty:
        return []
    n = min(len(s), max(int(min_holdings), math.ceil(len(s) * (1 - top_q))))
    return list(s.sort_values(ascending=False).index[:n])


def run_backtest(panel: pd.DataFrame, fwd: pd.DataFrame, universe: list[str], benchmark: str | None,
                 score_col: str, top_q: float = 0.9, holding: int = 1, cost_bps: float = 10.0,
                 min_names: int = 30, min_holdings: int = 10, lag: int = 0,
                 missing_return: str | float = "cash", eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly backtest on an unbroken calendar of month-ends.

    The series starts at the first month with a usable signal and runs, month by month,
    until the last month with forward returns; it never skips a calendar month. Rebalances
    are scheduled by elapsed months (every `holding` months from the start). If a scheduled
    month has no usable signal (fewer than `min_names` scores), the current holdings are
    carried until the next scheduled rebalance, and the row is marked `carried`.

    missing_return: what a holding earns in a month with no return.
    "cash" = 0% (default); "loss" = -100%, a worst case for delistings; or a number.
    """
    fill = float({"cash": 0.0, "loss": -1.0}.get(missing_return, missing_return))
    calendar = pd.date_range(fwd.index.min(), fwd.index.max(), freq="ME")
    fwd = fwd.reindex(calendar)
    scores = panel.pivot(index="month_end", columns="ticker", values=score_col).reindex(calendar)
    if lag:
        # The signal from month-end t is traded at t + lag.
        scores = scores.shift(int(lag))
    valid = (scores.notna().sum(axis=1) >= min_names).to_numpy()
    univ_fwd = fwd.reindex(columns=universe)
    if eligible is not None:   # equal-weight benchmark: index members at each date only
        univ_fwd = univ_fwd.where(eligible.reindex(index=calendar, columns=universe).fillna(True).astype(bool))
    has_fwd = (univ_fwd.notna().sum(axis=1) > 0).to_numpy()
    first = next((i for i in range(len(calendar)) if valid[i] and has_fwd[i]), None)
    if first is None:
        return pd.DataFrame()

    rows, cur, formed_at = [], None, None
    for i in range(first, len(calendar)):
        if not has_fwd[i]:
            break                       # no returns at all this month: end here rather than leave a gap
        d = calendar[i]
        scheduled = (i - first) % holding == 0
        traded, rebalanced = 0.0, False
        if scheduled and valid[i]:
            picks = select_top(scores.iloc[i], top_q, min_holdings)
            if picks:
                w = pd.Series(1.0 / len(picks), index=picks)
                prev = cur if cur is not None else pd.Series(dtype=float)
                idx = w.index.union(prev.index)
                traded = float((w.reindex(idx, fill_value=0.0) - prev.reindex(idx, fill_value=0.0)).abs().sum())
                cur, formed_at, rebalanced = w, i, True
        if cur is None:
            break
        r = fwd.iloc[i].reindex(cur.index)
        n_missing = int(r.isna().sum())
        r = r.fillna(fill)
        gross = float((cur * r).sum())
        rows.append({
            "formation_date": calendar[formed_at], "signal_date": calendar[max(formed_at - int(lag), 0)],
            "period_end": d + pd.offsets.MonthEnd(1), "gross": gross, "net": gross - traded * cost_bps / 1e4,
            "turnover": traded, "rebalanced": rebalanced, "carried": scheduled and not rebalanced,
            "n_holdings": len(cur), "n_missing": n_missing, "ew_universe": float(univ_fwd.iloc[i].mean()),
            "benchmark": float(fwd.iloc[i][benchmark]) if benchmark in fwd.columns else np.nan,
        })
        cur = cur * (1 + r)
        cur = cur / cur.sum() if cur.sum() > 0 else cur
    return pd.DataFrame(rows)


def assert_contiguous(period_ends: pd.Series) -> None:
    """Performance statistics assume one observation per calendar month, with no gaps."""
    pe = pd.DatetimeIndex(pd.to_datetime(pd.Series(period_ends)))
    if len(pe) and not pe.equals(pd.date_range(pe[0], periods=len(pe), freq="ME")):
        raise ValueError("Backtest returns are not consecutive month-ends; refusing to annualize a gapped series")


# --------------------------------------------------------------------------- statistics
def nw_lags(n: int) -> int:
    """Newey-West (1994) automatic bandwidth: floor(4 * (n / 100) ** (2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean with HAC (Bartlett-kernel) standard errors.

    Monthly strategy returns and overlapping-horizon ICs are serially correlated, so
    the textbook mean / (sd / sqrt(n)) overstates confidence. lags = 0 reduces to it
    (up to the n vs n-1 divisor).
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    lags = nw_lags(n) if lags is None else int(lags)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def perf_stats(r: pd.Series, rf_annual: float = 0.0, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"months": n}
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    growth = (1 + r).cumprod()
    excess_rf = r - rf_m
    out = {
        "months": n,
        "cagr": float(growth.iloc[-1] ** (12 / n) - 1),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
        "sharpe": float(excess_rf.mean() / excess_rf.std(ddof=1) * np.sqrt(12)) if excess_rf.std(ddof=1) > 0 else np.nan,
        # The starting NAV of 1.0 is a peak too: a loss in month one is a drawdown.
        "max_drawdown": float((growth / growth.cummax().clip(lower=1.0) - 1).min()),
        "total_return": float(growth.iloc[-1] - 1),
    }
    if bench is not None:
        b = bench.reindex(r.index)
        active = (r - b).dropna()
        te = active.std(ddof=1)
        out.update({
            "excess_return_ann": float(active.mean() * 12),
            "tracking_error": float(te * np.sqrt(12)),
            "information_ratio": float(active.mean() / te * np.sqrt(12)) if te > 0 else np.nan,
            "hit_rate": float((active > 0).mean()),
            "t_stat_excess_iid": float(active.mean() / (te / np.sqrt(len(active)))) if te > 0 else np.nan,
            "t_stat_excess": newey_west_tstat(active),
        })
    return out


def summarize(bt: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    assert_contiguous(bt["period_end"])
    s = bt.set_index("period_end")
    rows = {
        "strategy_net": perf_stats(s["net"], rf_annual, s["ew_universe"]),
        "strategy_gross": perf_stats(s["gross"], rf_annual, s["ew_universe"]),
        "ew_universe": perf_stats(s["ew_universe"], rf_annual),
    }
    if s["benchmark"].notna().any():
        rows["benchmark"] = perf_stats(s["benchmark"], rf_annual)
        rows["strategy_net_vs_benchmark"] = perf_stats(s["net"], rf_annual, s["benchmark"])
    out = pd.DataFrame(rows).T
    out["avg_turnover_per_rebalance"] = np.nan
    out["avg_holdings"] = np.nan
    reb = bt[bt["turnover"] > 0]
    out.loc[["strategy_net", "strategy_gross"], "avg_turnover_per_rebalance"] = reb["turnover"].mean() if len(reb) else 0.0
    out.loc[["strategy_net", "strategy_gross"], "avg_holdings"] = bt["n_holdings"].mean()
    out["missing_return_share"] = np.nan
    out.loc[["strategy_net", "strategy_gross"], "missing_return_share"] = (
        bt["n_missing"].sum() / bt["n_holdings"].sum() if bt["n_holdings"].sum() else 0.0)
    out["rebalances"] = int(bt["rebalanced"].sum()) if "rebalanced" in bt else np.nan
    out["carried_rebalances"] = int(bt["carried"].sum()) if "carried" in bt else np.nan
    out["start"] = s.index.min().strftime("%Y-%m-%d")
    out["end"] = s.index.max().strftime("%Y-%m-%d")
    return out


def decile_returns(panel: pd.DataFrame, score_col: str, horizons=(1, 6, 12), n_bins: int = 10,
                   min_names: int = 30) -> pd.DataFrame:
    """Average forward return by score decile (10 = best), and excess vs the scored cross-section."""
    df = panel[panel[score_col].notna()].copy()
    counts = df.groupby("month_end")[score_col].transform("count")
    df = df[counts >= max(min_names, n_bins)]
    df["decile"] = df.groupby("month_end")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False) + 1)
    out = []
    for h in horizons:
        col = f"ret_fwd_{h}m"
        d = df[df[col].notna()]
        if d.empty:
            continue
        d = d.assign(excess=d[col] - d.groupby("month_end")[col].transform("mean"))
        by_date = d.groupby(["month_end", "decile"])[[col, "excess"]].mean()
        agg = by_date.groupby("decile").mean()
        agg.columns = ["avg_fwd_return", "avg_excess_return"]
        agg["horizon_months"] = h
        agg["n_dates"] = by_date.reset_index().groupby("decile")["month_end"].nunique()
        out.append(agg.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ic_series(panel: pd.DataFrame, score_col: str, fwd_col: str, min_obs: int = 20) -> pd.Series:
    """Spearman rank correlation between score and forward return, per month."""
    vals = {}
    for d, g in panel[["month_end", score_col, fwd_col]].dropna().groupby("month_end"):
        if len(g) >= min_obs:
            vals[d] = g[score_col].rank().corr(g[fwd_col].rank())
    return pd.Series(vals, name=f"ic_{score_col}_{fwd_col}", dtype=float)


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """Mean IC with a Newey-West t-stat; lags cover at least the h-1 months of label overlap."""
    ic = ic.dropna()
    if len(ic) < 3:
        return {"n_months": len(ic)}
    sd = ic.std(ddof=1)
    lags = max(horizon - 1, nw_lags(len(ic)))
    return {"n_months": len(ic), "mean_ic": float(ic.mean()), "ic_std": float(sd),
            "t_stat": newey_west_tstat(ic, lags),
            "t_stat_iid": float(ic.mean() / (sd / np.sqrt(len(ic)))) if sd > 0 else np.nan,
            "nw_lags": lags, "pct_positive": float((ic > 0).mean())}
