"""Annual fundamentals from yfinance, Financial Modeling Prep, Alpha Vantage or SEC EDGAR.

Every provider is normalised to the same columns (FUND_COLS) so the rest of the
pipeline does not care where the numbers came from. The parsers are pure
functions (payload in, DataFrame out) so they can be unit-tested offline.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from .db import table_df, upsert_df
from .logging_utils import log_fetch

FUND_COLS = ["ticker", "period_end", "source", "available_date", "available_date_is_estimated",
             "version_type", "accession", "revenue", "ebit", "ebitda", "net_income", "eps_diluted",
             "shares_diluted", "total_equity", "total_debt", "cash", "operating_cash_flow", "capex",
             "total_assets"]
VALUE_COLS = FUND_COLS[7:]
VERSIONED_SOURCES = {"sec"}          # parsers that emit their own filing-dated versions


class RateLimitError(RuntimeError):
    """Provider says we are out of quota; stop calling it for this run."""


class BudgetExhausted(RuntimeError):
    """This run's own request budget (fundamentals.max_requests_per_run) is used up."""


class RequestBudget:
    """Counts every HTTP attempt (retries included) against an optional cap."""

    def __init__(self, limit: int | None):
        self.limit = None if limit is None else int(limit)
        self.used = 0

    def take(self, n: int = 1) -> None:
        if self.limit is not None and self.used + n > self.limit:
            raise BudgetExhausted(f"request budget of {self.limit} reached")
        self.used += n


# --------------------------------------------------------------------------- helpers
def _num(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "None", "null", "NaN", "-"}:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finalize(rows: list[dict], ticker: str, source: str, lag_days: int) -> pd.DataFrame:
    """Build the normalised frame and fill/validate available_date."""
    if not rows:
        return pd.DataFrame(columns=FUND_COLS)
    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    df["source"] = source
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
    if "available_date" in df.columns:
        filed = pd.to_datetime(df["available_date"], errors="coerce")
    else:
        filed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    estimate = df["period_end"] + pd.Timedelta(days=int(lag_days))
    # A filing date before the period end is bad data: fall back to the lag estimate.
    valid = filed.notna() & (filed >= df["period_end"])
    df["available_date"] = filed.where(valid, estimate)
    df["available_date_is_estimated"] = (~valid).astype(int)
    if "version_type" not in df:
        df["version_type"] = "original"
    if "accession" not in df:
        df["accession"] = None
    for col in VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df = df.dropna(subset=VALUE_COLS, how="all")
    df = df.sort_values(["period_end", "available_date"]).drop_duplicates(
        ["ticker", "period_end", "available_date"], keep="last")
    df["period_end"] = df["period_end"].dt.strftime("%Y-%m-%d")
    df["available_date"] = df["available_date"].dt.strftime("%Y-%m-%d")
    return df[FUND_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- yfinance
YF_KEYS = {
    "revenue": ["TotalRevenue", "OperatingRevenue"],
    "ebit": ["EBIT", "OperatingIncome"],
    "ebitda": ["EBITDA", "NormalizedEBITDA"],
    "net_income": ["NetIncomeCommonStockholders", "NetIncome"],
    "eps_diluted": ["DilutedEPS"],
    "shares_diluted": ["DilutedAverageShares"],
    "total_equity": ["StockholdersEquity", "CommonStockEquity"],
    "total_debt": ["TotalDebt"],
    "cash": ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"],
    "shares_outstanding": ["OrdinarySharesNumber", "ShareIssued"],
    "da": ["DepreciationAndAmortization", "DepreciationAmortizationDepletion", "ReconciledDepreciation"],
    "operating_cash_flow": ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"],
    "capex": ["CapitalExpenditure"],
    "total_assets": ["TotalAssets"],
}


def _yf_lookup(frames: list[pd.DataFrame | None], keys: list[str], col: pd.Timestamp) -> float:
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cols = pd.to_datetime(frame.columns)
        hits = np.where(np.abs((cols - col).days) <= 7)[0]
        if len(hits) == 0:
            continue
        c = frame.columns[hits[0]]
        for key in keys:
            if key in frame.index:
                val = _num(frame.at[key, c])
                if not np.isnan(val):
                    return val
    return np.nan


def parse_yfinance(income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame | None,
                   ticker: str, lag_days: int) -> pd.DataFrame:
    """Parse yfinance get_*(pretty=False) statements (rows = items, columns = period ends)."""
    if income is None or income.empty:
        return pd.DataFrame(columns=FUND_COLS)
    rows = []
    for col in pd.to_datetime(income.columns):
        inc, bs, cf = [income], [balance], [cashflow, income]
        row = {"period_end": col}
        for field in ("revenue", "ebit", "ebitda", "net_income", "eps_diluted", "shares_diluted"):
            row[field] = _yf_lookup(inc, YF_KEYS[field], col)
        for field in ("total_equity", "total_debt", "cash"):
            row[field] = _yf_lookup(bs, YF_KEYS[field], col)
        if np.isnan(row["ebitda"]):
            row["ebitda"] = row["ebit"] + _yf_lookup(cf, YF_KEYS["da"], col)
        if np.isnan(row["shares_diluted"]):
            row["shares_diluted"] = _yf_lookup(bs, YF_KEYS["shares_outstanding"], col)
        row["operating_cash_flow"] = _yf_lookup([cashflow], YF_KEYS["operating_cash_flow"], col)
        row["capex"] = -_yf_lookup([cashflow], YF_KEYS["capex"], col)       # reported as a negative outflow
        row["total_assets"] = _yf_lookup(bs, YF_KEYS["total_assets"], col)
        rows.append(row)
    return _finalize(rows, ticker, "yfinance", lag_days)


def _fetch_yfinance(ticker: str, ctx: dict) -> tuple[pd.DataFrame, int]:
    import yfinance as yf

    t = yf.Ticker(ticker)
    budget = ctx.get("budget")
    if budget:
        budget.take(3)            # three statement requests, made inside the library
    inc = t.get_income_stmt(pretty=False, freq="yearly")
    bs = t.get_balance_sheet(pretty=False, freq="yearly")
    cf = t.get_cash_flow(pretty=False, freq="yearly")
    return parse_yfinance(inc, bs, cf, ticker, ctx["lag_days"]), 3


# --------------------------------------------------------------------------- FMP
def parse_fmp(income: list[dict], balance: list[dict], ticker: str, lag_days: int,
              cashflow: list[dict] | None = None) -> pd.DataFrame:
    """Parse FMP annual income-statement and balance-sheet-statement JSON lists.

    Handles both the `stable` field names (filingDate, epsDiluted) and the legacy
    v3 names (fillingDate, epsdiluted).
    """
    bs_by_date = {b.get("date"): b for b in (balance or [])}
    cf_by_date = {c.get("date"): c for c in (cashflow or [])}
    rows = []
    for inc in income or []:
        d = inc.get("date")
        bs = bs_by_date.get(d, {})
        cf = cf_by_date.get(d, {})
        filed = inc.get("filingDate") or inc.get("fillingDate") or (inc.get("acceptedDate") or "")[:10] or None
        ebit = _num(inc.get("ebit")) if "ebit" in inc else np.nan
        if np.isnan(ebit):
            ebit = _num(inc.get("operatingIncome"))
        eps = inc.get("epsDiluted", inc.get("epsdiluted"))
        rows.append({
            "period_end": d,
            "available_date": filed,
            "revenue": _num(inc.get("revenue")),
            "ebit": ebit,
            "ebitda": _num(inc.get("ebitda")),
            "net_income": _num(inc.get("netIncome")),
            "eps_diluted": _num(eps),
            "shares_diluted": _num(inc.get("weightedAverageShsOutDil")),
            "total_equity": _num(bs.get("totalStockholdersEquity")),
            "total_debt": _num(bs.get("totalDebt")),
            "cash": _num(bs.get("cashAndCashEquivalents")),
            "operating_cash_flow": _num(cf.get("operatingCashFlow")),
            "capex": -_num(cf.get("capitalExpenditure")),          # FMP reports capex as negative
            "total_assets": _num(bs.get("totalAssets")),
        })
    return _finalize(rows, ticker, "fmp", lag_days)


def _fetch_fmp(ticker: str, ctx: dict) -> tuple[pd.DataFrame, int]:
    base = ctx["cfg"]["fundamentals"]["fmp_base_url"].rstrip("/")
    key = ctx["api_key"]
    limit = ctx["cfg"]["fundamentals"].get("fmp_limit", 10)
    payloads = {}
    for ep in ("income-statement", "balance-sheet-statement", "cash-flow-statement"):
        if base.endswith("/stable"):
            url, params = f"{base}/{ep}", {"symbol": ticker, "period": "annual", "limit": limit, "apikey": key}
        else:  # legacy v3 style: /api/v3/income-statement/AAPL
            url, params = f"{base}/{ep}/{ticker}", {"period": "annual", "limit": limit, "apikey": key}
        data = _get_json(ctx["session"], url, params, on_request=ctx["on_request"], budget=ctx["budget"])
        if isinstance(data, dict):
            msg = data.get("Error Message") or data.get("error") or json.dumps(data)[:200]
            if "limit" in msg.lower():
                raise RateLimitError(msg)
            raise ValueError(f"FMP error for {ticker}: {msg}")
        payloads[ep] = data
    _save_raw(ctx, ticker, payloads)
    return parse_fmp(payloads["income-statement"], payloads["balance-sheet-statement"],
                     ticker, ctx["lag_days"], payloads["cash-flow-statement"]), 3


# --------------------------------------------------------------------------- Alpha Vantage
def _av_check(payload: dict, ticker: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Alpha Vantage payload for {ticker}")
    if "annualReports" in payload:
        return
    if "Note" in payload or "Information" in payload:
        raise RateLimitError(payload.get("Note") or payload.get("Information"))
    raise ValueError(f"Alpha Vantage error for {ticker}: {json.dumps(payload)[:200]}")


def parse_alphavantage(income: dict, balance: dict, ticker: str, lag_days: int,
                       cashflow: dict | None = None) -> pd.DataFrame:
    """Parse INCOME_STATEMENT, BALANCE_SHEET and (optionally) CASH_FLOW responses (values arrive as strings)."""
    _av_check(income, ticker)
    _av_check(balance, ticker)
    if cashflow is not None:
        _av_check(cashflow, ticker)
    bs_by_date = {b.get("fiscalDateEnding"): b for b in balance.get("annualReports", [])}
    cf_by_date = {c.get("fiscalDateEnding"): c for c in (cashflow or {}).get("annualReports", [])}
    rows = []
    for inc in income.get("annualReports", []):
        d = inc.get("fiscalDateEnding")
        bs = bs_by_date.get(d, {})
        debt = _num(bs.get("shortLongTermDebtTotal"))
        if np.isnan(debt):
            parts = [_num(bs.get("longTermDebt")), _num(bs.get("shortTermDebt"))]
            debt = np.nansum(parts) if not all(np.isnan(parts)) else np.nan
        rows.append({
            "period_end": d,
            "available_date": None,  # Alpha Vantage gives no filing date -> lag estimate
            "revenue": _num(inc.get("totalRevenue")),
            "ebit": _num(inc.get("ebit")),
            "ebitda": _num(inc.get("ebitda")),
            "net_income": _num(inc.get("netIncome")),
            "eps_diluted": np.nan,   # derived later as net_income / shares
            "shares_diluted": _num(bs.get("commonStockSharesOutstanding")),
            "total_equity": _num(bs.get("totalShareholderEquity")),
            "total_debt": debt,
            "cash": _num(bs.get("cashAndCashEquivalentsAtCarryingValue")),
            "operating_cash_flow": _num(cf_by_date.get(d, {}).get("operatingCashflow")),
            "capex": _num(cf_by_date.get(d, {}).get("capitalExpenditures")),   # positive in Alpha Vantage
            "total_assets": _num(bs.get("totalAssets")),
        })
    return _finalize(rows, ticker, "alphavantage", lag_days)


def _fetch_alphavantage(ticker: str, ctx: dict) -> tuple[pd.DataFrame, int]:
    url = "https://www.alphavantage.co/query"
    payloads = {}
    for fn in ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"):
        payloads[fn] = _get_json(ctx["session"], url, {"function": fn, "symbol": ticker, "apikey": ctx["api_key"]},
                                 on_request=ctx["on_request"], budget=ctx["budget"])
        _av_check(payloads[fn], ticker)
    _save_raw(ctx, ticker, payloads)
    return parse_alphavantage(payloads["INCOME_STATEMENT"], payloads["BALANCE_SHEET"],
                              ticker, ctx["lag_days"], payloads["CASH_FLOW"]), 3


# --------------------------------------------------------------------------- SEC EDGAR
SEC_CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "revenue": [("Revenues", "USD"), ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
                ("SalesRevenueNet", "USD")],
    "ebit": [("OperatingIncomeLoss", "USD")],
    "da": [("DepreciationDepletionAndAmortization", "USD"), ("DepreciationAndAmortization", "USD"),
           ("DepreciationAmortizationAndAccretionNet", "USD")],
    "net_income": [("NetIncomeLoss", "USD")],
    "eps_diluted": [("EarningsPerShareDiluted", "USD/shares")],
    "shares_diluted": [("WeightedAverageNumberOfDilutedSharesOutstanding", "shares")],
    "total_equity": [("StockholdersEquity", "USD"),
                     ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")],
    "debt_total": [("LongTermDebt", "USD")],
    "debt_noncurrent": [("LongTermDebtNoncurrent", "USD")],
    "debt_current": [("LongTermDebtCurrent", "USD"), ("DebtCurrent", "USD")],
    "short_term_borrowings": [("ShortTermBorrowings", "USD")],
    "cash": [("CashAndCashEquivalentsAtCarryingValue", "USD"),
             ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "USD")],
    "operating_cash_flow": [("NetCashProvidedByUsedInOperatingActivities", "USD"),
                            ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD")],
    "capex": [("PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
              ("PaymentsToAcquireProductiveAssets", "USD")],
    "total_assets": [("Assets", "USD")],
}
SEC_FLOWS = {"revenue", "ebit", "da", "net_income", "eps_diluted", "shares_diluted", "operating_cash_flow", "capex"}


def _sec_entries(facts: dict, concept: str, unit: str, flow: bool) -> dict[str, list[dict]]:
    """{period_end: [{filed, val, accn, form}, ...]} for annual facts reported in 10-K / 10-K/A filings."""
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    except (KeyError, TypeError):
        return {}
    out: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("form") not in ("10-K", "10-K/A") or "end" not in e or "filed" not in e:
            continue
        if flow:
            if "start" not in e:
                continue
            days = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
            if not 350 <= days <= 380:
                continue
        out.setdefault(e["end"], []).append(
            {"filed": e["filed"], "val": _num(e.get("val")), "accn": e.get("accn"), "form": e["form"]})
    return out


def _as_of(entries: list[dict], filed: str) -> float:
    """Value from the latest filing on or before `filed` (NaN if none yet)."""
    known = [e for e in entries if e["filed"] <= filed]
    return max(known, key=lambda e: e["filed"])["val"] if known else np.nan


def _same(a: dict, b: dict, fields: list[str]) -> bool:
    for f in fields:
        x, y = a[f], b[f]
        if (np.isnan(x) and np.isnan(y)) or (not np.isnan(x) and not np.isnan(y)
                                             and abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))):
            continue
        return False
    return True


def parse_sec_companyfacts(facts: dict, ticker: str, lag_days: int) -> pd.DataFrame:
    """One row per (fiscal year, filing that changed what was known about it).

    The first row for a period is the original 10-K. A later 10-K/A that changes a
    value becomes an 'amendment' version, and a later 10-K whose comparative column
    restates the prior year becomes a 'restatement' version. Each version's
    available_date is its filing date, so a backtest only sees a revision after it
    was published.
    """
    series: dict[str, dict[str, list[dict]]] = {}
    for field, candidates in SEC_CONCEPTS.items():
        merged: dict[str, list[dict]] = {}
        for concept, unit in candidates:  # earlier candidates take priority per period
            for end, ents in _sec_entries(facts, concept, unit, field in SEC_FLOWS).items():
                merged.setdefault(end, ents)
        series[field] = merged

    # Fiscal year-ends are defined by annual-duration income-statement facts;
    # balance-sheet instants are only kept on those dates.
    fy_ends = sorted(set(series["net_income"]) | set(series["revenue"]))
    out_fields = ["revenue", "ebit", "ebitda", "net_income", "eps_diluted", "shares_diluted",
                  "total_equity", "total_debt", "cash", "operating_cash_flow", "capex", "total_assets"]
    rows = []
    for end in fy_ends:
        ents = {f: series[f].get(end, []) for f in SEC_CONCEPTS}
        filings: dict[str, dict] = {}
        for f_ents in ents.values():
            for e in f_ents:
                rec = filings.setdefault(e["filed"], {"accn": e["accn"], "amended": False})
                rec["amended"] |= e["form"] == "10-K/A"
        prev = None
        for filed in sorted(filings):
            v = {f: _as_of(ents[f], filed) for f in SEC_CONCEPTS}
            if np.isnan(v["net_income"]) and np.isnan(v["revenue"]):
                continue
            if not np.isnan(v["debt_total"]):
                debt = v["debt_total"] + np.nan_to_num(v["short_term_borrowings"])
            elif not np.isnan(v["debt_noncurrent"]):
                debt = v["debt_noncurrent"] + np.nan_to_num(v["debt_current"]) + np.nan_to_num(v["short_term_borrowings"])
            else:
                debt = np.nan
            row = {"revenue": v["revenue"], "ebit": v["ebit"], "ebitda": v["ebit"] + v["da"],
                   "net_income": v["net_income"], "eps_diluted": v["eps_diluted"],
                   "shares_diluted": v["shares_diluted"], "total_equity": v["total_equity"],
                   "total_debt": debt, "cash": v["cash"], "operating_cash_flow": v["operating_cash_flow"],
                   "capex": v["capex"], "total_assets": v["total_assets"]}
            if prev is not None and _same(row, prev, out_fields):
                continue
            kind = "original" if prev is None else ("amendment" if filings[filed]["amended"] else "restatement")
            rows.append({"period_end": end, "available_date": filed, "version_type": kind,
                         "accession": filings[filed]["accn"], **row})
            prev = row
    return _finalize(rows, ticker, "sec", lag_days)


def edgar_filing_url(cik: str | int, accession: str) -> str:
    """Folder of an EDGAR filing, e.g. 0000320193-20-000096 -> .../data/320193/000032019320000096/."""
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"


def _sec_cik_map(ctx: dict) -> dict[str, str]:
    if ctx.get("cik_map") is None:
        data = _get_json(ctx["session"], "https://www.sec.gov/files/company_tickers.json", {},
                         headers=ctx["sec_headers"], on_request=ctx["on_request"], budget=ctx["budget"])
        mapping = {}
        for rec in data.values():
            tk = str(rec["ticker"]).upper()
            cik = str(rec["cik_str"]).zfill(10)
            mapping[tk] = cik
            mapping[tk.replace(".", "-")] = cik
        ctx["cik_map"] = mapping
    return ctx["cik_map"]


def _fetch_sec(ticker: str, ctx: dict) -> tuple[pd.DataFrame, int]:
    calls = 0
    cik = ctx["universe_cik"].get(ticker)
    if not cik:
        calls += 0 if ctx.get("cik_map") is not None else 1
        cik = _sec_cik_map(ctx).get(ticker)
    if not cik:
        raise ValueError(f"No SEC CIK found for {ticker}")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = _get_json(ctx["session"], url, {}, headers=ctx["sec_headers"], on_request=ctx["on_request"], budget=ctx["budget"])
    _save_raw(ctx, ticker, facts)
    return parse_sec_companyfacts(facts, ticker, ctx["lag_days"]), calls + 1


# --------------------------------------------------------------------------- HTTP + cache
def _get_json(session: requests.Session, url: str, params: dict, headers: dict | None = None,
              retries: int = 3, on_request: Callable[[str, str, str], None] | None = None,
              budget: RequestBudget | None = None):
    """GET with retries. Every attempt is reported to `on_request(endpoint, status, message)`.

    `url` never contains credentials (API keys travel in `params`), so it is safe to log.
    """
    log = on_request or (lambda *_: None)
    for attempt in range(1, retries + 1):
        tag = f"attempt {attempt}/{retries}"
        if budget is not None:
            budget.take()             # raises BudgetExhausted before the request is sent
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            log(url, "request_failed", f"{tag}: {type(exc).__name__}")
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            continue
        code = resp.status_code
        if code == 429 or (code >= 500 and attempt < retries):
            final = attempt == retries
            log(url, "request_failed" if final else "request_retry", f"HTTP {code}, {tag}")
            if final:
                raise RateLimitError(f"HTTP 429 from {url}")
            time.sleep(2 ** attempt * (2 if code == 429 else 1))
            continue
        if code >= 400:
            log(url, "request_failed", f"HTTP {code}, {tag}")
            resp.raise_for_status()
        log(url, "request_ok", f"HTTP {code}, {tag}")
        return resp.json()
    raise RuntimeError(f"Failed to fetch {url}")


def _save_raw(ctx: dict, ticker: str, payload) -> None:
    raw = Path(ctx["raw_dir"]) / f"{ctx['source']}_raw"
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw / f"{ticker}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


FETCHERS: dict[str, tuple[Callable, str | None]] = {
    "yfinance": (_fetch_yfinance, None),
    "fmp": (_fetch_fmp, "FMP_API_KEY"),
    "alphavantage": (_fetch_alphavantage, "ALPHAVANTAGE_API_KEY"),
    "sec": (_fetch_sec, None),
}


def fetch_fundamentals(cfg: dict, tickers: list[str], conn: sqlite3.Connection, run_id: str,
                       logger: logging.Logger) -> dict:
    fcfg = cfg["fundamentals"]
    source = fcfg["source"]
    if source not in FETCHERS:
        raise ValueError(f"Unknown fundamentals source {source!r}; choose from {sorted(FETCHERS)}")
    fetcher, key_env = FETCHERS[source]
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(f"Set the {key_env} environment variable (or put it in .env) to use {source}")
    if source == "sec" and "example.com" in fcfg.get("sec_user_agent", "example.com"):
        raise RuntimeError("Set fundamentals.sec_user_agent to your name and email; SEC rejects anonymous clients")

    raw_dir = Path(cfg["project"]["raw_dir"])
    cache_dir = raw_dir / "fundamentals" / source
    cache_dir.mkdir(parents=True, exist_ok=True)
    secs = table_df(conn, "SELECT ticker, cik FROM securities")
    ctx = {
        "cfg": cfg, "source": source, "api_key": api_key, "raw_dir": str(raw_dir),
        "lag_days": int(fcfg["filing_lag_days"]), "session": requests.Session(),
        "sec_headers": {"User-Agent": fcfg.get("sec_user_agent", "")},
        "universe_cik": {r.ticker: r.cik for r in secs.itertuples() if isinstance(r.cik, str) and r.cik.strip("0")},
        "cik_map": None,
        "current_ticker": None,
    }
    ctx["budget"] = RequestBudget(fcfg.get("max_requests_per_run"))
    ctx["on_request"] = lambda endpoint, status, msg: log_fetch(
        conn, run_id, source, endpoint, ctx["current_ticker"], status, 0, msg)
    max_age = timedelta(days=int(fcfg.get("cache_days", 25)))
    pause = float(fcfg.get("request_pause_seconds", 0.25))
    blocked: str | bool = False
    counts = {"ok": 0, "cached": 0, "error": 0, "skipped": 0, "empty": 0}
    frames = []

    for tk in tickers:
        cache = cache_dir / f"{tk}.csv"
        cached_df = _read_cache(cache)                 # None if missing or unreadable (e.g. zero bytes)
        fresh = cached_df is not None and datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime) < max_age
        if fresh or (blocked and cached_df is not None):
            frames.append(cached_df)
            counts["cached"] += 1
            log_fetch(conn, run_id, source, "cache", tk, "cached", len(frames[-1]),
                      "stale cache (provider blocked)" if not fresh else "")
            continue
        if blocked:
            counts["skipped"] += 1
            log_fetch(conn, run_id, source, "fundamentals", tk, "skipped", 0, blocked)
            continue
        ctx["current_ticker"] = tk
        try:
            df, _ = fetcher(tk, ctx)
        except BudgetExhausted as exc:
            blocked = "max_requests_per_run reached earlier in run"
            counts["skipped"] += 1
            log_fetch(conn, run_id, source, "fundamentals", tk, "budget_exhausted", 0, str(exc))
            logger.warning("Request budget (%s) reached at %s; using cache only for the rest of this run",
                           ctx["budget"].limit, tk)
            continue
        except RateLimitError as exc:
            blocked = "rate limited earlier in run"
            counts["skipped"] += 1
            log_fetch(conn, run_id, source, "fundamentals", tk, "rate_limited", 0, str(exc))
            logger.warning("%s rate limit hit at %s; using cache only for the rest of this run", source, tk)
            continue
        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            log_fetch(conn, run_id, source, "fundamentals", tk, "error", 0, repr(exc))
            logger.warning("Fundamentals failed for %s: %r", tk, exc)
            time.sleep(pause)
            continue
        time.sleep(pause)
        status = "ok" if len(df) else "empty"
        counts[status] += 1
        log_fetch(conn, run_id, source, "fundamentals", tk, status, len(df))
        df.to_csv(cache, index=False)
        frames.append(df)

    nonempty = [f for f in frames if not f.empty]   # a provider can return nothing for every ticker
    combined = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=FUND_COLS)
    for col in FUND_COLS:            # caches written before versioning lack the new columns
        if col not in combined:
            combined[col] = "original" if col == "version_type" else None
    combined["version_type"] = combined["version_type"].fillna("original")
    n_revised = 0
    if len(combined):
        if combined.duplicated(["ticker", "period_end", "source", "available_date"]).any():
            raise RuntimeError("duplicate fundamental keys (ticker, period_end, source, available_date)")
        if source not in VERSIONED_SOURCES:
            combined = version_against_store(conn, combined[FUND_COLS], source)
            n_revised = int((combined["version_type"] == "revision_observed").sum())
        upsert_df(conn, "fundamentals", combined[FUND_COLS])
    calls = ctx["budget"].used
    logger.info("Fundamentals (%s): %s; %d rows written (%d observed revisions); %d requests",
                source, counts, len(combined), n_revised, calls)
    return {"rows": len(combined), "calls": calls, "revisions": n_revised, **counts}


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Cached normalised frame, or None when absent or unreadable (zero bytes, truncated)."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype={"ticker": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def version_against_store(conn: sqlite3.Connection, new: pd.DataFrame, source: str,
                          today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Turn a provider snapshot into versions instead of overwriting history.

    yfinance, FMP and Alpha Vantage return today's figures, which can include
    restatements made after the original filing. If a period is already stored and
    its values changed, the new values are added as a 'revision_observed' version
    dated today, so earlier months keep using what was stored before. Unchanged
    periods are dropped from the write. (The first download itself cannot be
    unwound: it may already contain restated values.)
    """
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    stored = table_df(conn, "SELECT * FROM fundamentals WHERE source = ? ORDER BY available_date", (source,))
    if stored.empty:
        return new
    latest = stored.groupby(["ticker", "period_end"]).tail(1).set_index(["ticker", "period_end"])
    keep = []
    for _, row in new.iterrows():
        key = (row["ticker"], row["period_end"])
        if key not in latest.index:
            keep.append(row)
            continue
        old = latest.loc[key]
        if _same({c: _num(row[c]) for c in VALUE_COLS}, {c: _num(old[c]) for c in VALUE_COLS}, VALUE_COLS):
            continue
        row = row.copy()
        row["available_date"] = max(today, pd.Timestamp(row["available_date"])).strftime("%Y-%m-%d")
        row["available_date_is_estimated"] = 0
        row["version_type"] = "revision_observed"
        keep.append(row)
    return pd.DataFrame(keep, columns=new.columns) if keep else pd.DataFrame(columns=new.columns)
