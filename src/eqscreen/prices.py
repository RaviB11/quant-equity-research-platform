"""Month-end prices and split history from Yahoo Finance (yfinance)."""
from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from .db import upsert_df
from .logging_utils import log_fetch


def last_complete_month_end(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last calendar month-end strictly before the current month (2026-11-12 -> 2026-10-31)."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    return today.to_period("M").to_timestamp() - pd.Timedelta(days=1)


def _field(data: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Return a date x ticker frame for one OHLC field from a yf.download result."""
    if isinstance(data.columns, pd.MultiIndex):
        if field not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index)
        out = data[field]
    else:  # single ticker, flat columns
        if field not in data.columns:
            return pd.DataFrame(index=data.index)
        out = data[[field]].rename(columns={field: tickers[0]})
    if isinstance(out, pd.Series):
        out = out.to_frame(tickers[0])
    return out


def daily_to_monthly(daily: pd.DataFrame, tickers: list[str],
                     include_partial_month: bool = False,
                     today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a yf.download(auto_adjust=False, actions=True) frame to long monthly prices + splits."""
    daily = daily.copy()
    idx = pd.to_datetime(daily.index)
    daily.index = idx.tz_localize(None) if idx.tz is not None else idx
    close = _field(daily, "Close", tickers)
    adj = _field(daily, "Adj Close", tickers)
    if adj.empty:
        adj = close
    m_close = close.resample("ME").last()
    m_adj = adj.resample("ME").last()

    if not include_partial_month:
        cutoff = last_complete_month_end(today)
        m_close = m_close.loc[m_close.index <= cutoff]
        m_adj = m_adj.loc[m_adj.index <= cutoff]

    long_close = m_close.stack().rename("close")
    long_adj = m_adj.stack().rename("adj_close")
    monthly = pd.concat([long_close, long_adj], axis=1).dropna(subset=["close"]).reset_index()
    monthly.columns = ["month_end", "ticker", "close", "adj_close"]
    monthly["month_end"] = monthly["month_end"].dt.strftime("%Y-%m-%d")

    splits = _field(daily, "Stock Splits", tickers)
    if splits.empty:
        split_long = pd.DataFrame(columns=["ticker", "split_date", "ratio"])
    else:
        s = splits.stack()
        s = s[s.fillna(0) > 0].rename("ratio").reset_index()
        s.columns = ["split_date", "ticker", "ratio"]
        s["split_date"] = pd.to_datetime(s["split_date"]).dt.strftime("%Y-%m-%d")
        split_long = s[["ticker", "split_date", "ratio"]]
    return monthly[["ticker", "month_end", "close", "adj_close"]], split_long


def fetch_prices(cfg: dict, tickers: list[str], conn: sqlite3.Connection, run_id: str,
                 logger: logging.Logger) -> dict:
    import yfinance as yf  # imported lazily so demo mode and tests do not need network libs

    pcfg = cfg["prices"]
    all_tickers = list(dict.fromkeys(tickers + [pcfg["benchmark"]]))
    batch_size = int(pcfg.get("batch_size", 100))
    n_rows, n_splits, missing = 0, 0, []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        endpoint = f"yf.download[{len(batch)} tickers] start={pcfg['start']}"
        try:
            data = yf.download(batch, start=pcfg["start"], auto_adjust=False, actions=True,
                               group_by="column", progress=False, threads=True,
                               multi_level_index=True)
        except Exception as exc:  # noqa: BLE001
            log_fetch(conn, run_id, "yfinance", endpoint, ",".join(batch[:5]), "error", 0, repr(exc))
            logger.error("Price batch %d failed: %r", i // batch_size + 1, exc)
            missing.extend(batch)
            continue
        if data is None or data.empty:
            log_fetch(conn, run_id, "yfinance", endpoint, ",".join(batch[:5]), "empty", 0)
            missing.extend(batch)
            continue

        monthly, splits = daily_to_monthly(data, batch, pcfg.get("include_partial_month", False))
        got = set(monthly["ticker"].unique())
        for tk in batch:
            status = "ok" if tk in got else "empty"
            log_fetch(conn, run_id, "yfinance", "prices", tk, status,
                      int((monthly["ticker"] == tk).sum()))
            if tk not in got:
                missing.append(tk)
        n_rows += upsert_df(conn, "prices_monthly", monthly)
        n_splits += upsert_df(conn, "splits", splits)
        logger.info("Prices batch %d: %d tickers, %d month-end rows, %d splits",
                    i // batch_size + 1, len(got), len(monthly), len(splits))

    if missing:
        logger.warning("No price data for %d tickers: %s", len(missing), ", ".join(missing[:20]))
    return {"rows": n_rows, "splits": n_splits, "missing": missing}
