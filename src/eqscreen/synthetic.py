"""Synthetic market used to test the pipeline end to end without network access.

THE DATA IS FAKE. It exists to check the machinery, not to say anything about
real markets. It has known properties the tests can verify:

* A persistent latent "cheapness" variable drives both reported earnings yield /
  book-to-market and next-month returns, so value factors SHOULD show a positive
  information coefficient and a positive top-minus-universe spread.
* No momentum effect is planted, so momentum SHOULD come out near zero.
* A few tickers have stock splits, with fundamentals reported in pre-split units
  (like real filings) and prices split-adjusted (like Yahoo), so split
  adjustment is exercised.
* Filing dates lag fiscal year-end by 40-85 days, so point-in-time logic matters.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .db import upsert_df
from .fundamentals import FUND_COLS
from .logging_utils import utc_now

SECTORS = ["Information Technology", "Health Care", "Financials", "Industrials", "Consumer Discretionary",
           "Consumer Staples", "Energy", "Communication Services", "Utilities"]


def generate(n_tickers: int = 150, start: str = "2009-01-31", end: str = "2025-12-31",
             seed: int = 7, value_premium: float = 0.004) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="ME")
    T, N = len(months), n_tickers
    tickers = [f"SYN{i:03d}" for i in range(N)]

    # Latent cheapness: AR(1) with unit stationary variance.
    phi = 0.98
    latent = np.zeros((T, N))
    latent[0] = rng.standard_normal(N)
    for t in range(1, T):
        latent[t] = phi * latent[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(N)

    mkt = rng.normal(0.007, 0.04, T)
    beta = rng.uniform(0.7, 1.3, N)
    idio = rng.normal(0.0, 0.06, (T, N))
    rets = np.zeros((T, N))
    rets[1:] = beta * mkt[1:, None] + value_premium * latent[:-1] + idio[1:]
    prices = rng.uniform(20, 200, N) * np.cumprod(1 + rets, axis=0)   # split-adjusted series

    # Staggered listings: ~10% of names start trading later.
    start_idx = np.zeros(N, dtype=int)
    late = rng.choice(N, size=N // 10, replace=False)
    start_idx[late] = rng.integers(12, T // 2, size=len(late))
    for i in range(N):
        prices[: start_idx[i], i] = np.nan

    shares_now = rng.uniform(1e8, 5e9, N)     # current (post-split) share count

    # Splits on four tickers, mid-sample.
    split_rows = []
    for i in rng.choice(N, size=4, replace=False):
        when = months[rng.integers(max(start_idx[i] + 24, 36), T - 12)]
        split_rows.append({"ticker": tickers[i], "split_date": when.strftime("%Y-%m-%d"),
                           "ratio": float(rng.choice([2.0, 3.0, 4.0]))})
    splits = pd.DataFrame(split_rows, columns=["ticker", "split_date", "ratio"])

    sectors = rng.choice(SECTORS, size=N)
    securities = pd.DataFrame({"ticker": tickers, "name": [f"Synthetic Co {i:03d}" for i in range(N)],
                               "sector": sectors, "sub_industry": None, "cik": None,
                               "updated_at": utc_now()})

    price_rows = pd.DataFrame({
        "ticker": np.repeat(tickers, T),
        "month_end": np.tile(months.strftime("%Y-%m-%d"), N),
        "close": prices.T.ravel(),
    })
    price_rows["adj_close"] = price_rows["close"]       # no dividends in the synthetic world
    price_rows = price_rows.dropna(subset=["close"])

    bench = 100 * np.cumprod(1 + mkt + rng.normal(0, 0.002, T))
    bench_rows = pd.DataFrame({"ticker": "SPY", "month_end": months.strftime("%Y-%m-%d"),
                               "close": bench, "adj_close": bench})

    # Annual fundamentals, December fiscal year-end, reported in as-filed (pre-split) units.
    fund_rows = []
    dec_idx = [k for k, m in enumerate(months) if m.month == 12]
    for i, tk in enumerate(tickers):
        tk_splits = splits[splits["ticker"] == tk]
        for k in dec_idx:
            if np.isnan(prices[k, i]) or rng.random() < 0.03:
                continue
            period_end = months[k]
            filed = period_end + pd.Timedelta(days=int(rng.integers(40, 86)))
            later = tk_splits[pd.to_datetime(tk_splits["split_date"]) > filed]["ratio"]
            factor = float(np.prod(later)) if len(later) else 1.0     # splits not yet happened at filing
            mcap = prices[k, i] * shares_now[i]
            ey = 0.06 + 0.025 * latent[k, i] + rng.normal(0, 0.01)
            eps_adj = prices[k, i] * ey
            net_income = eps_adj * shares_now[i]
            ebit = net_income * rng.uniform(1.2, 1.4)
            bm = np.exp(-1.0 + 0.3 * latent[k, i] + rng.normal(0, 0.3))
            fund_rows.append({
                "ticker": tk,
                "period_end": period_end.strftime("%Y-%m-%d"),
                "source": "synthetic",
                "available_date": filed.strftime("%Y-%m-%d"),
                "available_date_is_estimated": 0,
                "version_type": "original",
                "accession": None,
                "revenue": ebit / rng.uniform(0.08, 0.30),
                "ebit": ebit,
                "ebitda": ebit * rng.uniform(1.15, 1.40) if rng.random() > 0.05 else np.nan,
                "net_income": net_income,
                "eps_diluted": eps_adj * factor,                 # as reported, pre-split units
                "shares_diluted": shares_now[i] / factor,        # as reported, pre-split units
                "total_equity": mcap * bm,
                "total_debt": mcap * rng.uniform(0.05, 0.50),
                "cash": mcap * rng.uniform(0.02, 0.15),
                # Cash-flow items carry no planted return signal: earnings-quality ICs should be ~0.
                "operating_cash_flow": net_income * rng.uniform(0.6, 1.5),
                "capex": abs(ebit) * rng.uniform(0.1, 0.6),
                "total_assets": mcap * rng.uniform(0.6, 2.0),
            })
    fundamentals = pd.DataFrame(fund_rows, columns=FUND_COLS)

    # Index history: 12 current members joined mid-sample, each replacing a company that is
    # absent from the data (as in reality, where removed names are usually missing).
    joiners = rng.choice(N, size=12, replace=False)
    changes = pd.DataFrame([{
        "change_date": months[rng.integers(max(start_idx[i] + 12, 24), T - 24)] - pd.Timedelta(days=10),
        "added": tickers[i], "added_name": f"Synthetic Co {i:03d}",
        "removed": f"GONE{k:03d}", "removed_name": f"Departed Co {k:03d}",
    } for k, i in enumerate(joiners)])

    return {"securities": securities, "prices": pd.concat([price_rows, bench_rows], ignore_index=True),
            "splits": splits, "fundamentals": fundamentals, "index_changes": changes}


def load_into_db(conn: sqlite3.Connection, n_tickers: int = 150, seed: int = 7) -> dict:
    from .universe import store_membership

    data = generate(n_tickers=n_tickers, seed=seed)
    for table in ("securities", "prices_monthly", "splits", "fundamentals"):
        conn.execute(f"DELETE FROM {table}")
    store_membership(conn, data["securities"]["ticker"].tolist(), data["index_changes"])
    upsert_df(conn, "securities", data["securities"])
    upsert_df(conn, "prices_monthly", data["prices"])
    upsert_df(conn, "splits", data["splits"])
    upsert_df(conn, "fundamentals", data["fundamentals"])
    return {k: len(v) for k, v in data.items()}
