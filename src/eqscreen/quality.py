"""Data quality checks. Results are stored in the quality_checks table each run.

`fail` stops the pipeline (when quality.fail_on_error is true); `warn` is logged
and shown in the report so a human can decide whether it matters.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .db import table_df
from .logging_utils import utc_now


class DataQualityError(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail
    value: float | None
    threshold: float | None
    detail: str = ""


def _integrity(df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """(duplicate rows, non-month-end dates, first offending tickers)."""
    dup = df.duplicated(["ticker", "month_end"], keep=False)
    bad_date = df["month_end"] != df["month_end"] + pd.offsets.MonthEnd(0)
    names = sorted(df.loc[dup | bad_date, "ticker"].unique())[:10]
    return int(df.duplicated(["ticker", "month_end"]).sum()), int(bad_date.sum()), names


def _ratio_check(name, value, threshold, detail, warn_only=False) -> Check:
    ok = value >= threshold
    return Check(name, "pass" if ok else ("warn" if warn_only else "fail"), float(value), float(threshold), detail)


def run_checks(cfg: dict, conn: sqlite3.Connection, today: pd.Timestamp | None = None) -> list[Check]:
    q = cfg["quality"]
    source = cfg["fundamentals"]["source"]
    bench = cfg["prices"]["benchmark"]
    checks: list[Check] = []

    secs = table_df(conn, "SELECT ticker, sector FROM securities")
    prices = table_df(conn, "SELECT ticker, month_end, close, adj_close FROM prices_monthly", date_cols=("month_end",))
    all_fund = table_df(conn, "SELECT * FROM fundamentals WHERE source = ?", (source,),
                        date_cols=("period_end", "available_date"))
    splits = table_df(conn, "SELECT * FROM splits", date_cols=("split_date",))
    n_univ = len(secs)
    # Every check below describes the CURRENT universe. The database can hold records for
    # tickers from earlier universes; they must not make coverage or freshness look better.
    universe = set(secs["ticker"])
    fund = all_fund[all_fund["ticker"].isin(universe)]
    bench_prices = prices[prices["ticker"] == bench]

    # Integrity. Records this run uses (current universe + benchmark) must be sound, or the run
    # fails. Records retained for tickers outside the universe are history this run does not
    # use: problems there are reported as a warning, not a reason to stop.
    used_tickers = universe | {bench}
    used = prices[prices["ticker"].isin(used_tickers)]
    retained = prices[~prices["ticker"].isin(used_tickers)]
    dup_secs = int(secs["ticker"].duplicated().sum())
    u_dup, u_dates, u_names = _integrity(used)
    checks.append(Check("data_integrity", "fail" if dup_secs or u_dup or u_dates else "pass", u_dup + u_dates + dup_secs, 0,
                        f"universe + benchmark: {dup_secs} duplicate securities, {u_dup} duplicate price rows, "
                        f"{u_dates} non-month-end price dates" + (f" ({', '.join(u_names)})" if u_names else "")))
    r_dup, r_dates, r_names = _integrity(retained)
    checks.append(Check("retained_records_integrity", "warn" if r_dup or r_dates else "pass", r_dup + r_dates, 0,
                        f"outside the universe (not used by this run): {r_dup} duplicate price rows, {r_dates} "
                        "non-month-end dates" + (f" ({', '.join(r_names)})" if r_names else "")))
    prices = prices[prices["month_end"] == prices["month_end"] + pd.offsets.MonthEnd(0)]
    bench_prices = prices[prices["ticker"] == bench]

    checks.append(_ratio_check("universe_size", n_univ, q["min_universe_size"], f"{n_univ} securities"))
    # A "top decile" of a small universe is only a handful of names; the backtest then
    # holds min_holdings instead, i.e. a larger slice than the label suggests.
    b = cfg["backtest"]
    decile_n = int(np.ceil(n_univ * (1 - float(b["top_quantile"]))))
    breadth_ok = decile_n >= int(b["min_holdings"])
    checks.append(Check("universe_breadth", "pass" if breadth_ok else "warn", decile_n, b["min_holdings"],
                        f"top slice = {decile_n} names" + ("" if breadth_ok else
                        f"; backtest holds {b['min_holdings']} of {n_univ} instead. Treat as a smoke test, "
                        "not as evidence about the factors")))

    univ_prices = prices[prices["ticker"].isin(secs["ticker"])]
    counts = univ_prices.groupby("ticker").size()
    covered = int((counts >= 13).sum())
    cov = covered / n_univ if n_univ else 0.0
    checks.append(_ratio_check("price_coverage", cov, q["min_price_coverage"],
                               f"{covered}/{n_univ} tickers have 13+ monthly prices"))

    bad_px = int((univ_prices["close"] <= 0).sum() + univ_prices["close"].isna().sum())
    checks.append(Check("price_positive", "pass" if bad_px == 0 else "fail", bad_px, 0,
                        f"{bad_px} non-positive or missing closes"))

    ref = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    expected = ref.to_period("M").to_timestamp() - pd.Timedelta(days=1)
    demo_note = " (synthetic data has a fixed end date)" if cfg["mode"] == "demo" else ""
    if len(univ_prices):
        last_by_ticker = univ_prices.groupby("ticker")["month_end"].max()
        latest = last_by_ticker.max()
        stale_days = (expected - latest).days
        status = "pass" if stale_days <= q["max_price_staleness_days"] else "warn" if cfg["mode"] == "demo" else "fail"
        checks.append(Check("price_staleness_days", status, stale_days, q["max_price_staleness_days"],
                            f"latest universe month-end {latest:%Y-%m-%d}, expected {expected:%Y-%m-%d}" + demo_note))
        # One fresh ticker must not hide many stale ones.
        behind = last_by_ticker[last_by_ticker < latest]
        share = len(behind) / n_univ if n_univ else 0.0
        thr = float(q.get("max_stale_ticker_share", 0.10))
        checks.append(Check("stale_tickers", "pass" if share <= thr else "warn", share, thr,
                            f"{len(behind)}/{n_univ} universe tickers end before {latest:%Y-%m-%d}"
                            + (f": {', '.join(sorted(behind.index)[:10])}" if len(behind) else "")))
    else:
        checks.append(Check("price_staleness_days", "fail", None, q["max_price_staleness_days"],
                            "no prices for any ticker in the current universe"))

    wide = univ_prices.pivot(index="month_end", columns="ticker", values="adj_close").sort_index()
    rets = wide.pct_change(fill_method=None)
    extreme = rets.abs() > q["extreme_monthly_return"]
    n_extreme = int(extreme.sum().sum())
    names = sorted(extreme.any()[extreme.any()].index)[:10]
    checks.append(Check("extreme_monthly_returns", "pass" if n_extreme == 0 else "warn", n_extreme, 0,
                        f"|monthly return| > {q['extreme_monthly_return']:.0%}: {', '.join(names) or 'none'}"))

    if len(bench_prices) and len(univ_prices):
        b_last, u_last = bench_prices["month_end"].max(), univ_prices["month_end"].max()
        b_ok = b_last >= u_last
        checks.append(Check("benchmark_present", "pass" if b_ok else "warn", float(len(bench_prices)), 1,
                            f"{bench}: {len(bench_prices)} month-end rows, last {b_last:%Y-%m-%d}"
                            + ("" if b_ok else f" (universe runs to {u_last:%Y-%m-%d})")))
    else:
        checks.append(Check("benchmark_present", "warn", float(len(bench_prices)), 1, f"no {bench} prices"))

    with_fund = fund["ticker"].nunique()
    fcov = with_fund / n_univ if n_univ else 0.0
    old = all_fund.loc[~all_fund["ticker"].isin(universe), "ticker"].nunique()
    checks.append(_ratio_check("fundamental_coverage", fcov, q["min_fundamental_coverage"],
                               f"{with_fund}/{n_univ} universe tickers have {source} fundamentals"
                               + (f" ({old} tickers outside the universe ignored)" if old else "")))

    if len(fund):
        bad_dates = int((fund["available_date"] < fund["period_end"]).sum())
        checks.append(Check("available_after_period_end", "pass" if bad_dates == 0 else "fail", bad_dates, 0,
                            "fundamentals must not be available before their period ends"))
        est = float(fund["available_date_is_estimated"].mean())
        checks.append(Check("filing_dates_estimated_share", "pass", est, None,
                            f"{est:.0%} of records use period_end + {cfg['fundamentals']['filing_lag_days']}d "
                            "instead of a real filing date"))
        latest = fund.sort_values("available_date").groupby(["ticker", "period_end"]).tail(1)
        latest = latest.merge(secs, on="ticker", how="left")
        latest = latest[~latest["sector"].isin(cfg["factors"].get("ev_excluded_sectors", []))]
        if len(latest):
            ev_ok = float((latest["total_debt"].notna() & latest["cash"].notna()).mean())
            thr = float(q.get("min_ev_input_coverage", 0.8))
            checks.append(Check("ev_input_coverage", "pass" if ev_ok >= thr else "warn", ev_ok, thr,
                                f"{ev_ok:.0%} of non-financial statements report both debt and cash; the rest get "
                                f"no EV factors (policy: {cfg['factors'].get('ev_missing_policy', 'exclude')})"))
        from .factors import split_adjust_enabled
        checks.append(_share_discontinuity(fund, splits, split_adjust_enabled(cfg)))
        n_periods = fund.groupby(["ticker", "period_end"]).ngroups
        revised = int((fund.groupby(["ticker", "period_end"]).size() > 1).sum())
        kinds = fund.loc[fund["version_type"] != "original", "version_type"].value_counts().to_dict()
        checks.append(Check("revised_periods", "pass", revised, None,
                            f"{revised}/{n_periods} fiscal periods have more than one version {kinds or ''}".strip()))
    return checks


def _share_discontinuity(fund: pd.DataFrame, splits: pd.DataFrame, split_adjust: bool) -> Check:
    """Flag tickers whose split-adjusted share count jumps >1.6x between annual reports.

    A jump usually means a split the adjustment missed, or a provider that already
    restates history (in which case set fundamentals.split_adjust: false).
    """
    from .factors import split_factor

    f = (fund.dropna(subset=["shares_diluted"]).sort_values(["ticker", "period_end", "available_date"])
         .groupby(["ticker", "period_end"]).head(1).copy())       # original version of each period
    if f.empty:
        return Check("share_count_discontinuity", "pass", 0, 0, "no share data")
    factor = split_factor(f, splits, "available_date") if split_adjust else 1.0
    f["shares_adj"] = f["shares_diluted"] * factor
    ratio = f.groupby("ticker")["shares_adj"].transform(lambda s: s / s.shift(1))
    flagged = sorted(f.loc[(ratio > 1.6) | (ratio < 1 / 1.6), "ticker"].unique())
    return Check("share_count_discontinuity", "pass" if not flagged else "warn", len(flagged), 0,
                 f"tickers: {', '.join(flagged[:10]) or 'none'}")


def persist_and_enforce(cfg: dict, conn: sqlite3.Connection, run_id: str, checks: list[Check],
                        logger: logging.Logger) -> pd.DataFrame:
    now = utc_now()
    conn.executemany(
        "INSERT INTO quality_checks (run_id, checked_at_utc, check_name, status, value, threshold, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(run_id, now, c.name, c.status, None if c.value is None or np.isnan(c.value) else c.value,
          c.threshold, c.detail) for c in checks],
    )
    conn.commit()
    for c in checks:
        level = logging.INFO if c.status == "pass" else logging.WARNING if c.status == "warn" else logging.ERROR
        logger.log(level, "QC %-30s %-4s %s", c.name, c.status.upper(), c.detail)
    failed = [c for c in checks if c.status == "fail"]
    if failed and cfg["quality"].get("fail_on_error", True):
        raise DataQualityError("Data quality checks failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed))
    return pd.DataFrame([c.__dict__ for c in checks])
