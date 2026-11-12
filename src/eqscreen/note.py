"""One-page analyst note for a single stock, built only from the pipeline's own data.

    python -m eqscreen.note --ticker AAPL                 # latest scored month, live database
    python -m eqscreen.note --ticker SYN005 --mode demo   # synthetic, for checking the template
    python -m eqscreen.note --ticker AAPL --as-of 2024-06

Every figure comes from the factor panel, the point-in-time statement version used
that month, month-end prices and the fetch log. Nothing is estimated or typed in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import perf_stats
from .config import load_config
from .db import connect, table_df
from .factors import load_panel, price_panel
from .fundamentals import edgar_filing_url


def _f(x, kind="num", d=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return "n/a"
    if kind == "pct":
        return f"{x:.{d}%}"
    if kind == "spct":
        return f"{x:+.{d}%}"
    if kind == "money":
        return f"{x / 1e9:,.{d}f} bn" if abs(x) >= 1e8 else f"{x / 1e6:,.{d}f} m"
    return f"{x:,.{d}f}"


def _md_table(rows: list[list[str]], header: list[str], text_cols: tuple[int, ...] = (0,)) -> str:
    align = ["---" if i in text_cols else "---:" for i in range(len(header))]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_note(cfg: dict, ticker: str, as_of: str | None = None) -> str:
    conn = connect(cfg["project"]["db_path"])
    panel = load_panel(conn)
    t = ticker.upper()
    mine = panel[(panel["ticker"] == t) & panel["composite"].notna()]
    if mine.empty:
        raise ValueError(f"{t} has no composite score in {cfg['project']['db_path']}")
    if as_of:
        target = pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)
        mine = mine[mine["month_end"] <= target]
        if mine.empty:
            raise ValueError(f"{t} has no composite score on or before {target:%Y-%m-%d}")
    row = mine.sort_values("month_end").iloc[-1]
    m = row["month_end"]
    cross = panel[(panel["month_end"] == m) & panel["composite"].notna()]
    peers = cross[cross["sector"] == row["sector"]]
    rank = int((cross["composite"] > row["composite"]).sum()) + 1

    # The exact statement version the screen used that month (point in time).
    stmt = table_df(conn, "SELECT * FROM fundamentals WHERE ticker = ? AND source = ? AND period_end = ? "
                          "AND available_date = ?",
                    (t, cfg["fundamentals"]["source"], row["period_end"].strftime("%Y-%m-%d"),
                     row["available_date"].strftime("%Y-%m-%d")))
    s = stmt.iloc[0] if len(stmt) else pd.Series(dtype=object)
    versions = table_df(conn, "SELECT period_end, available_date, version_type, accession, available_date_is_estimated "
                              "FROM fundamentals WHERE ticker = ? AND source = ? AND available_date <= ? "
                              "ORDER BY period_end DESC, available_date", (t, cfg["fundamentals"]["source"],
                                                                         m.strftime("%Y-%m-%d")))
    sec = table_df(conn, "SELECT name, cik FROM securities WHERE ticker = ?", (t,))
    last_fetch = table_df(conn, "SELECT MAX(fetched_at_utc) AS ts FROM fetch_log WHERE ticker = ? AND status IN "
                                "('ok', 'cached', 'request_ok')", (t,))["ts"].iloc[0]

    # Price-based risk over the 36 months to the note date.
    px = price_panel(conn)
    r = px[t].pct_change(fill_method=None).loc[:m].dropna().tail(36)
    bench = cfg["prices"]["benchmark"]
    rb = px[bench].pct_change(fill_method=None).reindex(r.index) if bench in px else pd.Series(dtype=float)
    risk = perf_stats(r) if len(r) >= 12 else {}
    beta = (r.cov(rb) / rb.var()) if len(rb.dropna()) >= 12 and rb.var() > 0 else np.nan
    conn.close()

    synthetic = cfg["mode"] == "demo"
    lines = []
    if synthetic:
        lines += ["> **SYNTHETIC DEMO DATA.** This note was generated from a simulated market to check the template. "
                  "It describes no real company.", ""]
    lines += [f"# {row['name'] or t} ({t}): screening note", "",
              f"Screen month-end: **{m:%Y-%m-%d}**. Sector: {row['sector']}. Fundamentals: "
              f"{cfg['fundamentals']['source']}." + (f" Data last fetched {last_fetch} UTC." if last_fetch else ""), "",
              "## Where it sits in the screen", "",
              f"**{row['label']}**: ranked {rank} of {len(cross)} scored stocks (percentile "
              f"{_f(row['composite_pct'], 'pct', 0)}). Composite {_f(row['composite'], d=2)}, built from value "
              f"{_f(row['value_score'], d=2)} and momentum {_f(row['momentum_score'], d=2)} "
              f"(quality {_f(row['quality_score'], d=2)}, not weighted by default). Scores are z-scores: 0 is the "
              "month's average stock, 1 is one standard deviation better.", ""]

    def med(df, c):
        return df[c].median() if c in df and df[c].notna().any() else np.nan

    vals = [("P/E", "pe", "num"), ("P/B", "pb", "num"), ("EV/EBITDA", "ev_ebitda", "num"), ("EV/EBIT", "ev_ebit", "num"),
            ("PEG", "peg", "num"), ("Earnings yield", "earnings_yield", "pct"), ("12-1 momentum", "momentum_12_1", "spct")]
    lines += ["## Valuation against peers", "",
              _md_table([[n, _f(row[c], k), _f(med(peers, c), k), _f(med(cross, c), k)] for n, c, k in vals],
                        ["Measure", t, f"Sector median (n={len(peers)})", f"Universe median (n={len(cross)})"]), "",
              "Medians use stocks scored the same month. Blank EV multiples mean the sector is excluded or debt "
              "or cash was not reported.", ""]

    if len(s):
        ebit_margin = s["ebit"] / s["revenue"] if pd.notna(s.get("revenue")) and s["revenue"] else np.nan
        net_margin = s["net_income"] / s["revenue"] if pd.notna(s.get("revenue")) and s["revenue"] else np.nan
        prof = [["Revenue", _f(s["revenue"], "money")], ["EBIT", _f(s["ebit"], "money")],
                ["EBIT margin", _f(ebit_margin, "pct")], ["Net income", _f(s["net_income"], "money")],
                ["Net margin", _f(net_margin, "pct")], ["EPS growth vs prior year", _f(row["eps_growth"], "spct")],
                ["Return on equity", _f(row["roe"], "pct")],
                ["Operating cash flow", _f(s.get("operating_cash_flow"), "money")],
                ["Capex", _f(s.get("capex"), "money")], ["Free cash flow", _f(row.get("fcf"), "money")],
                ["FCF yield (FCF / market cap)", _f(row.get("fcf_yield"), "pct")],
                ["Cash conversion (OCF / net income)", _f(row.get("cash_conversion"), d=2)],
                ["Accruals ratio", _f(row.get("accruals_ratio"), "pct")],
                ["Total debt", _f(s["total_debt"], "money")],
                ["Cash", _f(s["cash"], "money")], ["Market cap", _f(row["market_cap"], "money")]]
        lines += [f"## Financial profile (fiscal year to {row['period_end']:%Y-%m-%d})", "",
                  _md_table(prof, ["Item", "Value"]), "",
                  "Accruals ratio = (net income − operating cash flow) / average total assets. Negative or low "
                  "values mean profits are backed by cash; high values mean more of the profit is accounting "
                  "accruals, which research associates with weaker future earnings. It can mislead for fast "
                  "growers (working capital builds up) and around acquisitions or one-off items.", "",
                  f"This is the version of the statement available on {m:%Y-%m-%d}: {s['version_type']}, public from "
                  f"{row['available_date']:%Y-%m-%d}" + (" (estimated filing date)" if s["available_date_is_estimated"]
                                                        else "") + ".", ""]

    hist = panel[(panel["ticker"] == t) & (panel["month_end"] <= m)].sort_values("month_end").tail(25)
    hist = hist[hist["composite_pct"].notna()].iloc[::6]
    if len(hist):
        lines += ["## Screen history", "",
                  _md_table([[d.strftime("%Y-%m"), _f(p, "pct", 0), lab] for d, p, lab in
                             zip(hist["month_end"], hist["composite_pct"], hist["label"])],
                            ["Month", "Percentile", "Label"], text_cols=(0, 2)), ""]

    if risk:
        lines += ["## Price risk (trailing 36 months, monthly total returns)", "",
                  _md_table([["Annualized volatility", _f(risk.get("ann_vol"), "pct")],
                             ["Return, annualized", _f(risk.get("cagr"), "pct")],
                             ["Max drawdown", _f(risk.get("max_drawdown"), "pct")],
                             [f"Beta vs {bench}", _f(beta, d=2)]], ["Measure", "Value"]), ""]

    if len(versions):
        cik = sec["cik"].iloc[0] if len(sec) and isinstance(sec["cik"].iloc[0], str) else None
        recent = versions[versions["period_end"].isin(sorted(versions["period_end"].unique())[-2:])]
        rows = []
        for v in recent.itertuples():
            link = (f"[{v.accession}]({edgar_filing_url(cik, v.accession)})" if cik and isinstance(v.accession, str)
                    else (v.accession if isinstance(v.accession, str) else "provider snapshot, no accession"))
            rows.append([v.period_end, v.available_date + (" (est.)" if v.available_date_is_estimated else ""),
                         v.version_type, link])
        lines += ["## Filings behind these numbers", "",
                  _md_table(rows, ["Fiscal year end", "Public from", "Version", "Filing"], text_cols=(0, 1, 2, 3)), ""]

    lines += ["## Assumptions and limits", "",
              "- Annual statements only; figures are as published by the date shown, including later amendments "
              "only after they were filed.",
              "- Peer and universe comparisons use today's index members (survivorship bias in any history).",
              f"- EV = market cap + total debt − cash; EV factors are omitted when debt or cash is missing "
              f"(policy: {cfg['factors'].get('ev_missing_policy', 'exclude')}).",
              "- The label is a mechanical screening classification for prioritizing research, not a rating, "
              "recommendation or price target.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="One-page screening note for a ticker")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--mode", choices=["live", "demo"], default="live")
    ap.add_argument("--as-of", default=None, help="YYYY-MM; default is the latest scored month")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, mode=args.mode)
    text = build_note(cfg, args.ticker, args.as_of)
    out = Path(cfg["project"]["output_dir"]) / "notes"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.ticker.upper()}_{(args.as_of or 'latest').replace('-', '')}.md"
    path.write_text(text, encoding="utf-8")
    print(f"Note written to {path}")


if __name__ == "__main__":
    main()
