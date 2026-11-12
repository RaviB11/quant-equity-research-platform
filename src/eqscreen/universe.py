"""Investable universe: current S&P 500 constituents (Wikipedia) or a fixed CSV."""
from __future__ import annotations

import logging
import sqlite3
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .db import upsert_df
from .logging_utils import log_fetch, utc_now

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_COLS = ["ticker", "name", "sector", "sub_industry", "cik"]


def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia lists class shares with a dot (BRK.B); Yahoo uses a dash (BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def parse_wikipedia_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), attrs={"id": "constituents"})
    if not tables:
        raise ValueError("Constituents table not found on the Wikipedia page")
    raw = tables[0]
    rename = {"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
              "GICS Sub-Industry": "sub_industry", "CIK": "cik"}
    missing = [c for c in rename if c not in raw.columns]
    if missing:
        raise ValueError(f"Wikipedia table layout changed; missing columns {missing}")
    df = raw[list(rename)].rename(columns=rename)
    df["ticker"] = df["ticker"].map(to_yahoo_symbol)
    df["cik"] = df["cik"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(10)
    return df.drop_duplicates("ticker").reset_index(drop=True)


def _clean_symbol(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    sym = str(x).strip()
    return to_yahoo_symbol(sym) if sym and sym.lower() != "nan" else None


def parse_wikipedia_changes(html: str) -> pd.DataFrame:
    """Parse the "Selected changes to the list of S&P 500 components" table."""
    tables = pd.read_html(StringIO(html), attrs={"id": "changes"})
    if not tables:
        raise ValueError("Changes table not found on the Wikipedia page")
    t = tables[0]
    flat = []
    for c in t.columns:
        parts = c if isinstance(c, tuple) else (c,)
        uniq = list(dict.fromkeys(str(x) for x in parts if not str(x).startswith("Unnamed")))
        flat.append(" ".join(uniq).strip())
    t.columns = flat

    def find(*words):
        for c in flat:
            if all(w in c.lower() for w in words):
                return c
        raise ValueError(f"Wikipedia changes table layout changed; no column with {words}")
    out = pd.DataFrame({
        "change_date": pd.to_datetime(t[find("date")].astype(str).str.replace(r"\[.*?\]", "", regex=True),
                                      errors="coerce"),
        "added": t[find("added", "ticker")].map(_clean_symbol),
        "added_name": t[find("added", "security")],
        "removed": t[find("removed", "ticker")].map(_clean_symbol),
        "removed_name": t[find("removed", "security")],
    })
    return out.dropna(subset=["change_date"]).reset_index(drop=True)


def reconstruct_membership(current: list[str], changes: pd.DataFrame) -> pd.DataFrame:
    """Membership spans by walking back from today's list through dated additions/removals.

    A stock added on date d is a member from d; a stock removed on d was a member until
    d - 1 day. Spans that reach back past the earliest recorded change have a NULL start
    (membership before that date is not documented by the source).
    """
    open_end: dict[str, pd.Timestamp | None] = {t: None for t in current}
    spans = []
    for date, grp in sorted(changes.groupby("change_date"), key=lambda kv: kv[0], reverse=True):
        for t in grp["added"].dropna():
            if t in open_end:
                spans.append((t, date, open_end.pop(t)))
        for t in grp["removed"].dropna():
            if t not in open_end:
                open_end[t] = date - pd.Timedelta(days=1)
    spans += [(t, None, end) for t, end in open_end.items()]
    out = pd.DataFrame(spans, columns=["ticker", "start_date", "end_date"])
    for c in ("start_date", "end_date"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d").where(out[c].notna(), None)
    return out.sort_values(["ticker", "start_date"], na_position="first").reset_index(drop=True)


def store_membership(conn: sqlite3.Connection, current: list[str], changes: pd.DataFrame) -> pd.DataFrame:
    spans = reconstruct_membership(current, changes)
    conn.execute("DELETE FROM index_changes")
    conn.execute("DELETE FROM membership")
    ch = changes.copy()
    ch["change_date"] = pd.to_datetime(ch["change_date"]).dt.strftime("%Y-%m-%d")
    upsert_df(conn, "index_changes", ch[["change_date", "added", "added_name", "removed", "removed_name"]])
    upsert_df(conn, "membership", spans)
    return spans


def fetch_sp500_wikipedia(raw_dir: str, conn: sqlite3.Connection, run_id: str,
                          logger: logging.Logger) -> pd.DataFrame:
    headers = {"User-Agent": "eqscreen/0.1 (research project; python-requests)"}
    try:
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        df = parse_wikipedia_table(resp.text)
    except Exception as exc:  # noqa: BLE001 - logged and re-raised
        log_fetch(conn, run_id, "wikipedia", WIKI_SP500_URL, None, "error", 0, repr(exc))
        raise
    out = Path(raw_dir) / f"universe_sp500_{date.today():%Y%m%d}.csv"
    df.to_csv(out, index=False)
    log_fetch(conn, run_id, "wikipedia", WIKI_SP500_URL, None, "ok", len(df), f"saved {out.name}")
    logger.info("Universe: %d S&P 500 constituents from Wikipedia (saved %s)", len(df), out.name)
    try:
        changes = parse_wikipedia_changes(resp.text)
        changes.to_csv(Path(raw_dir) / f"sp500_changes_{date.today():%Y%m%d}.csv", index=False)
        spans = store_membership(conn, df["ticker"].tolist(), changes)
        logger.info("Index changes: %d dated changes back to %s; %d membership spans reconstructed",
                    len(changes), changes["change_date"].min().date(), len(spans))
    except Exception as exc:  # noqa: BLE001 - membership is an enhancement, not a requirement
        log_fetch(conn, run_id, "wikipedia", WIKI_SP500_URL + "#changes", None, "error", 0, repr(exc))
        conn.execute("DELETE FROM membership")
        conn.execute("DELETE FROM index_changes")
        conn.commit()
        logger.warning("Could not parse S&P 500 change history (%r); backtest will use today's members only", exc)
    return df


def load_universe_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} needs at least a 'ticker' column")
    for col in UNIVERSE_COLS:
        if col not in df.columns:
            df[col] = None
    df["ticker"] = df["ticker"].map(to_yahoo_symbol)
    df["sector"] = df["sector"].fillna("Unknown")
    return df[UNIVERSE_COLS].drop_duplicates("ticker").reset_index(drop=True)


def build_universe(cfg: dict, conn: sqlite3.Connection, run_id: str,
                   logger: logging.Logger) -> pd.DataFrame:
    ucfg = cfg["universe"]
    # Membership history belongs to the source that produced it: clear it on every build, so a
    # CSV universe or a failed change-table parse never inherits a previous run's membership.
    n_old = conn.execute("SELECT COUNT(*) FROM membership").fetchone()[0]
    conn.execute("DELETE FROM membership")
    conn.execute("DELETE FROM index_changes")
    conn.commit()
    if n_old and ucfg["source"] != "sp500_wikipedia":
        logger.info("Cleared %d stored membership spans: the %s universe has no membership history", n_old,
                    ucfg["source"])
    if ucfg["source"] == "sp500_wikipedia":
        df = fetch_sp500_wikipedia(cfg["project"]["raw_dir"], conn, run_id, logger)
    elif ucfg["source"] == "file":
        df = load_universe_file(ucfg["file"])
        logger.info("Universe: %d tickers from %s", len(df), ucfg["file"])
    else:
        raise ValueError(f"Unknown universe source {ucfg['source']!r}")

    if ucfg.get("max_tickers"):
        df = df.head(int(ucfg["max_tickers"]))
        logger.info("Universe capped at %d tickers (universe.max_tickers)", len(df))

    if not df["ticker"].is_unique:
        raise ValueError(f"duplicate tickers in universe: {sorted(df.loc[df['ticker'].duplicated(), 'ticker'])[:10]}")
    df = df[UNIVERSE_COLS].copy()
    df["updated_at"] = utc_now()
    # Replace the securities table so the universe matches this run exactly.
    conn.execute("DELETE FROM securities")
    upsert_df(conn, "securities", df)
    return df


def survivorship_summary(conn: sqlite3.Connection) -> dict:
    """How much of the index's history is missing from the data (for disclosure)."""
    ch = pd.read_sql_query("SELECT * FROM index_changes", conn)
    if ch.empty:
        return {"membership_data": False}
    first = conn.execute("SELECT MIN(p.month_end) FROM prices_monthly p JOIN securities s ON s.ticker = p.ticker"
                         ).fetchone()[0]
    priced = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM prices_monthly")}
    since = ch[ch["change_date"] >= (first or "0000")]
    removed = since["removed"].dropna().unique()
    with_prices = [t for t in removed if t in priced]
    return {"membership_data": True, "earliest_recorded_change": ch["change_date"].min(),
            "first_price_month": first, "changes_since_first_price": int(len(since)),
            "removed_since_first_price": int(len(removed)), "removed_with_price_data": len(with_prices),
            "removed_missing_from_data": int(len(removed) - len(with_prices))}
