"""Analyst-ready research report for one company.

    python -m eqscreen.research --ticker AAPL --init-risks   # writes config/research/AAPL_risks.md to fill in
    python -m eqscreen.research --ticker AAPL                # writes outputs/research/AAPL_<yyyymm>.md

Combines: financial fundamentals (multi-year, point in time), peer valuation multiples, DCF
scenarios (if config/dcf/<TICKER>.yaml is filled in), earnings quality, factor rankings, price
risk, key risks (analyst-written file plus mechanical flags), source filings, assumptions and
limitations. Every number comes from the pipeline's own data; nothing is typed in by the code.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import dcf
from . import earnings_quality as eq
from .backtest import perf_stats
from .config import ROOT, load_config
from .db import connect, table_df
from .factors import load_panel, price_panel, split_factor
from .fundamentals import edgar_filing_url
from .note import _f, _md_table

RISKS_TEMPLATE = """<!-- Analyst-written key risks for {ticker}. Replace each prompt with your own assessment,
     citing filings (e.g. the 10-K "Risk Factors" section, with its date) or other sources.
     The research report stays DRAFT until this file has real text AND a filled-in
     "Reviewed by: <name>, <YYYY-MM-DD>" line below. -->

Reviewed by:

### Business and competitive
-

### Financial (leverage, liquidity, earnings quality)
-

### Regulatory, legal and governance
-

### Valuation and market
-
"""


def sign_off_date(text: str) -> pd.Timestamp | None:
    m = SIGN_OFF.search(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    return pd.Timestamp(m.group("date")) if m else None


SIGN_OFF = re.compile(r"^\s*Reviewed by:\s*(?P<who>[^,\n]*\w[^,\n]*),\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$", re.M)
CITATION = re.compile(r"https?://|\b10-[KQ]\b|\b8-K\b|\bsource:", re.I)


def risks_signed_off(text: str) -> bool:
    return bool(SIGN_OFF.search(re.sub(r"<!--.*?-->", "", text, flags=re.S)))


def risks_cite_sources(text: str) -> bool:
    return bool(CITATION.search(re.sub(r"<!--.*?-->", "", text, flags=re.S)))


def risks_have_text(text: str) -> bool:
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    body = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("#") and not line.strip().lower().startswith("reviewed by:"))
    return len(re.sub(r"[\s\-\*]", "", body)) >= 40


def risks_written(text: str) -> bool:
    """Real text (beyond the template) and a named, dated sign-off line."""
    return risks_have_text(text) and risks_signed_off(text)


def _pct_rank(series: pd.Series, value) -> float:
    s = series.dropna()
    return float((s <= value).mean()) if len(s) and pd.notna(value) else np.nan


def build_report(cfg: dict, ticker: str, as_of: str | None = None, assumptions_path: Path | None = None,
                 risks_path: Path | None = None) -> tuple[str, dict]:
    t = ticker.upper()
    th = {**eq.DEFAULT_FLAGS, "max_net_debt_to_ebitda": 3.0, "max_terminal_share": 0.75, "max_dcf_gap": 0.5,
          "max_drawdown_36m": -0.30, "high_beta": 1.3, **cfg.get("research", {}).get("flags", {})}
    n_peers = int(cfg.get("research", {}).get("peers", 6))
    conn = connect(cfg["project"]["db_path"])
    panel = load_panel(conn)
    mine = panel[(panel["ticker"] == t) & panel["composite"].notna()]
    if as_of:
        mine = mine[mine["month_end"] <= pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)]
    if mine.empty:
        conn.close()
        raise ValueError(f"{t} has no composite score{' on or before ' + as_of if as_of else ''}")
    row = mine.sort_values("month_end").iloc[-1]
    m = row["month_end"]
    cross = panel[(panel["month_end"] == m) & panel["composite"].notna()]
    sector = cross[cross["sector"] == row["sector"]]
    src = cfg["fundamentals"]["source"]
    sec_row = table_df(conn, "SELECT name, cik FROM securities WHERE ticker = ?", (t,))
    cik = sec_row["cik"].iloc[0] if len(sec_row) and isinstance(sec_row["cik"].iloc[0], str) else None

    # ---- multi-year fundamentals, point in time, split-consistent EPS
    f = table_df(conn, "SELECT * FROM fundamentals WHERE ticker = ? AND source = ? AND available_date <= ? "
                       "ORDER BY period_end, available_date", (t, src, m.strftime("%Y-%m-%d")),
                 date_cols=("period_end", "available_date"))
    f = f.groupby("period_end").tail(1).sort_values("period_end").tail(6).reset_index(drop=True)
    splits = table_df(conn, "SELECT * FROM splits", date_cols=("split_date",))
    f["eps_adj"] = f["eps_diluted"] / split_factor(f, splits, "available_date")
    hist = eq.history(conn, cfg, t, m)
    eqs = eq.summarize(hist, cross, t, cfg)

    # ---- price risk, data provenance
    px = price_panel(conn)
    r = px[t].pct_change(fill_method=None).loc[:m].dropna().tail(36)
    bench = cfg["prices"]["benchmark"]
    rb = px[bench].pct_change(fill_method=None).reindex(r.index) if bench in px else pd.Series(dtype=float)
    risk = perf_stats(r) if len(r) >= 12 else {}
    beta = r.cov(rb) / rb.var() if len(rb.dropna()) >= 12 and rb.var() > 0 else np.nan
    fetched = table_df(conn, "SELECT source, MAX(fetched_at_utc) AS last FROM fetch_log WHERE ticker = ? AND status IN "
                             "('ok', 'cached', 'request_ok') GROUP BY source", (t,))
    versions = table_df(conn, "SELECT period_end, available_date, version_type, accession, available_date_is_estimated "
                              "FROM fundamentals WHERE ticker = ? AND source = ? AND available_date <= ? "
                              "ORDER BY period_end DESC, available_date", (t, src, m.strftime("%Y-%m-%d")))
    conn.close()
    run_path = Path(cfg["project"]["output_dir"]) / "run_summary.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}

    # ---- DCF scenarios (optional)
    a_path = assumptions_path or ROOT / "config" / "dcf" / f"{t}.yaml"
    dcf_res, dcf_note, scen, assume, imp = None, None, None, None, None
    if a_path.exists():
        assume = yaml.safe_load(a_path.read_text(encoding="utf-8"))
        # One report, one date: a DCF is only included when its valuation month is the report
        # month, so a historical report can never pick up a later statement, price or beta.
        try:
            dcf_month = dcf.check_identity(assume, t)
        except dcf.AssumptionError as exc:
            dcf_month, dcf_note = None, f"DCF omitted: {exc}."
        if dcf_month is None:
            pass
        elif dcf_month != m:
            dcf_note = (f"DCF omitted: the assumptions file is dated {dcf_month:%Y-%m} but this report is as of "
                        f"{m:%Y-%m}. A DCF is shown only when both refer to the same month. Run the report with "
                        f"--as-of {dcf_month:%Y-%m}, or create assumptions dated {m:%Y-%m}.")
        else:
            try:
                base = dcf.load_base(cfg, t, m.strftime("%Y-%m"))
                if base["as_of"] != m:
                    raise dcf.AssumptionError(f"no statement-backed valuation base at {m:%Y-%m} "
                                              f"(latest is {base['as_of']:%Y-%m})")
                scen = dcf.value_scenarios(base, assume)
                dcf_res = dcf.value(base, dcf.scenario_assumptions(assume)["base"])
                dcf_res["as_of"] = base["as_of"]
                imp = dcf.implied_assumptions(base, dcf.scenario_assumptions(assume)["base"])
            except dcf.AssumptionError as exc:
                dcf_note = f"DCF not available: {exc}"
    else:
        dcf_note = (f"DCF not run: create {a_path.relative_to(ROOT) if a_path.is_relative_to(ROOT) else a_path} with "
                    f"`python -m eqscreen.dcf --ticker {t} --init` and fill in the assumptions.")

    # ---- risks
    r_path = risks_path or ROOT / "config" / "research" / f"{t}_risks.md"
    risks_text = r_path.read_text(encoding="utf-8") if r_path.exists() else ""
    draft = not risks_written(risks_text)
    tol = int(cfg.get("research", {}).get("contemporaneous_days", 45))
    signed = sign_off_date(risks_text)
    retro_risks = (not draft) and dcf.retrospective(signed, m, tol)
    retro_dcf = scen is not None and dcf.retrospective(assume.get("prepared_on"), m, tol)
    undated_dcf = scen is not None and not assume.get("prepared_on")

    # ---- mechanical flags
    flags = list(eqs["flags"])
    last = f.iloc[-1] if len(f) else pd.Series(dtype=float)
    ebitda = last.get("ebitda")
    net_debt = (last.get("total_debt") - last.get("cash")) if len(f) else np.nan
    nd_ebitda = net_debt / ebitda if pd.notna(net_debt) and pd.notna(ebitda) and ebitda > 0 else np.nan
    if pd.notna(nd_ebitda) and nd_ebitda > th["max_net_debt_to_ebitda"]:
        flags.append(f"Net debt is {nd_ebitda:.1f}x EBITDA (above {th['max_net_debt_to_ebitda']:.1f}x).")
    if dcf_res is not None and dcf_res["terminal_share"] > th["max_terminal_share"]:
        flags.append(f"Terminal value is {dcf_res['terminal_share']:.0%} of DCF enterprise value: the valuation rests "
                     "mostly on long-run assumptions.")
    if dcf_res is not None and abs(dcf_res["upside"]) > th["max_dcf_gap"]:
        flags.append(f"DCF base value differs from the share price by {dcf_res['upside']:+.0%}: either the market "
                     "disagrees strongly with the assumptions or an assumption needs another look.")
    if risk and risk.get("max_drawdown", 0) < th["max_drawdown_36m"]:
        flags.append(f"Share price fell {risk['max_drawdown']:.0%} from peak within the last 36 months.")
    if pd.notna(beta) and beta > th["high_beta"]:
        flags.append(f"Beta of {beta:.2f} against {bench}: more volatile than the market.")
    if row["available_date_is_estimated"]:
        flags.append("The latest statement has no recorded filing date; its availability is estimated.")
    if not row["ev_inputs_complete"]:
        flags.append("Debt or cash is missing from the latest statement, so EV-based multiples are blank.")
    if len(versions) and (versions[versions["period_end"] == row["period_end"].strftime("%Y-%m-%d")]
                          ["version_type"] != "original").any():
        flags.append("The latest fiscal year has been amended or restated since first filed.")
    if not draft and not risks_cite_sources(risks_text):
        flags.append("The key-risks section cites no sources (no filing, URL or 'Source:' reference).")
    if undated_dcf:
        flags.append("DCF assumptions are undated (no prepared_on); their timing relative to the report is unknown.")
    if scen is not None and dcf.missing_sources(assume):
        flags.append(f"{len(dcf.missing_sources(assume))} DCF input(s) have no recorded source: "
                     f"{', '.join(dcf.missing_sources(assume))}.")

    # ---------------------------------------------------------------- render
    rank = int((cross["composite"] > row["composite"]).sum()) + 1
    sec_rank = int((sector["composite"] > row["composite"]).sum()) + 1
    name = row["name"] or t
    price_src = "synthetic" if cfg["mode"] == "demo" else cfg["prices"].get("source", "yfinance")
    L = []
    if cfg["mode"] == "demo":
        L += ["> **SYNTHETIC DEMO DATA.** This report was generated from a simulated market to check the template. "
              "It describes no real company.", ""]
    if undated_dcf:
        L += ["> **Undated DCF assumptions.** The assumptions file records no `prepared_on` date, so the report "
              "cannot show whether the valuation was written at the report date or with hindsight.", ""]
    if retro_risks or retro_dcf:
        parts = ([f"key risks reviewed on {signed:%Y-%m-%d}"] if retro_risks else []) + \
                ([f"DCF assumptions written on {assume['prepared_on']}"] if retro_dcf else [])
        L += [f"> **Retrospective analysis.** The report's data are as of {m:%Y-%m-%d}, but the "
              f"{' and '.join(parts)} came later and may use information not available on that date.", ""]
    if draft:
        why = "have not been signed off" if risks_have_text(risks_text) else "have not been written yet"
        L += [f"> **DRAFT.** Key risks {why} (`{r_path.name}` needs real text and a "
              "'Reviewed by: name, YYYY-MM-DD' line).", ""]
    L += [f"# {name} ({t}): equity research report", "",
          f"Data as of month-end **{m:%Y-%m-%d}**. Sector: {row['sector']}. Fundamentals: {src}. "
          f"Prices: {price_src}. Run: {run.get('run_id', 'n/a')}.", "",
          "## Summary", ""]
    summ = [f"**Screen:** {row['label']}, ranked {rank} of {len(cross)} stocks and {sec_rank} of {len(sector)} in its "
            f"sector (composite percentile {_f(row['composite_pct'], 'pct', 0)})."]
    summ.append(f"**Relative valuation:** P/E {_f(row['pe'])} against a sector median of {_f(sector['pe'].median())}; "
                f"EV/EBIT {_f(row['ev_ebit'])} against {_f(sector['ev_ebit'].median())}.")
    if scen is not None:
        b = scen.set_index("scenario").loc["base"]
        summ.append(f"**DCF:** base case {b['value_per_share']:,.2f} per share against a price of {dcf_res['price']:,.2f} "
                    f"({b['upside']:+.1%}); scenario range {scen['value_per_share'].min():,.2f} to "
                    f"{scen['value_per_share'].max():,.2f}.")
    else:
        summ.append(f"**DCF:** {dcf_note}")
    summ.append(f"**Earnings quality:** {len(eqs['flags'])} flag(s); median cash conversion "
                f"{_f(eqs.get('median_cash_conversion'), d=2)}, FCF positive in {eqs.get('fcf_positive_years', 0)} of "
                f"{eqs.get('fcf_years', 0)} years.")
    summ.append(f"**Flags for review:** {len(flags)} (see Key risks).")
    L += [" ".join(summ), ""]

    if len(f):
        yrs = f["period_end"].dt.strftime("%Y-%m").tolist()

        def line(label, vals, kind="money", d=1):
            return [label] + [_f(v, kind, d) for v in vals]
        rev_g = f["revenue"] / f["revenue"].shift(1) - 1
        L += ["## Financial fundamentals", "",
              _md_table([line("Revenue", f["revenue"]), line("Revenue growth", rev_g, "spct"),
                         line("EBIT margin", f["ebit"] / f["revenue"], "pct"),
                         line("Net income", f["net_income"]), line("Diluted EPS (split-adjusted)", f["eps_adj"], "num", 2),
                         line("Return on equity", (f["net_income"] / f["total_equity"].where(f["total_equity"] > 0)), "pct"),
                         line("Total debt", f["total_debt"]), line("Cash", f["cash"]),
                         line("Net debt / EBITDA", (f["total_debt"] - f["cash"]) / f["ebitda"].where(f["ebitda"] > 0),
                              "num", 1)],
                        ["Fiscal year"] + yrs), "",
              "Each year shows the latest version of its statement published by the report date.", ""]

    peers = sector[sector["ticker"] != t].copy()
    if len(peers):
        peers["dist"] = (np.log(peers["market_cap"]) - np.log(row["market_cap"])).abs()
        peers = peers.nsmallest(n_peers, "dist")
    cols = [("Mkt cap", "market_cap", "money"), ("P/E", "pe", "num"), ("P/B", "pb", "num"),
            ("EV/EBITDA", "ev_ebitda", "num"), ("EV/EBIT", "ev_ebit", "num"), ("FCF yield", "fcf_yield", "pct"),
            ("Composite pct", "composite_pct", "pct")]
    prow = [[f"**{t}**"] + [_f(row[c], k) for _, c, k in cols]]
    prow += [[p.ticker] + [_f(getattr(p, c), k) for _, c, k in cols] for p in peers.itertuples()]
    prow += [[f"Sector median (n={len(sector)})"] + [_f(sector[c].median(), k) for _, c, k in cols]]
    L += ["## Peer valuation", "", f"The {len(peers)} sector peers closest in market cap, same month.", "",
          _md_table(prow, ["Company"] + [n for n, _, _ in cols]), ""]

    L += ["## DCF valuation", ""]
    if scen is not None:
        L += [f"Valuation month {dcf_res['as_of']:%Y-%m-%d}; WACC {dcf_res['wacc']:.2%} (base case). Full model with "
              f"forecast, bridge and sensitivity: `python -m eqscreen.dcf --ticker {t}`.", ""]
        L += dcf.render_scenarios(scen, dcf_res["price"]) or [
            f"Base case only: {dcf_res['value_per_share']:,.2f} per share ({dcf_res['upside']:+.1%}).", ""]
        L += [x.replace("## ", "### ", 1) if x.startswith("## ") else x
              for x in dcf.render_implied(imp, dcf.scenario_assumptions(assume)["base"])]
        L = [x.replace("## Scenarios", "### Scenarios") for x in L]
    else:
        L += [dcf_note, ""]

    L += ["## Earnings quality", ""]
    if len(hist):
        L += [_md_table([[d.strftime("%Y-%m"), _f(r_.operating_cash_flow, "money"), _f(r_.capex, "money"),
                          _f(r_.fcf, "money"), _f(r_.fcf_margin, "pct"), _f(r_.fcf_growth, "spct"),
                          _f(r_.cash_conversion, d=2), _f(r_.accruals_ratio, "pct")]
                         for d, r_ in zip(hist["period_end"], hist.itertuples())],
                        ["Fiscal year", "Operating cash flow", "Capex", "FCF", "FCF margin", "FCF growth",
                         "Cash conversion", "Accruals ratio"]), ""]
        extra = []
        if "fcf_cagr" in eqs:
            extra.append(f"FCF compound growth over the period: {eqs['fcf_cagr']:+.1%} a year.")
        if "accruals_percentile" in eqs:
            extra.append(f"Latest accruals ratio is higher than {eqs['accruals_percentile']:.0%} of the universe "
                         "that month (higher = less cash-backed).")
        L += [" ".join(extra) or "Not enough history for trend statistics.", "",
              "Cash conversion = operating cash flow ÷ net income; accruals ratio = (net income − operating cash flow) "
              "÷ average total assets. Meaning and limits: docs/earnings_quality.md.", ""]
    else:
        L += ["No statement history available.", ""]

    groups = [("Composite", "composite"), ("Value", "value_score"), ("Momentum", "momentum_score"),
              ("Quality (ROE)", "quality_score"), ("Earnings quality (low accruals)", "earnings_quality_score")]
    L += ["## Factor rankings", "",
          _md_table([[n, _f(row[c], d=2), _f(_pct_rank(cross[c], row[c]), "pct", 0),
                      _f(_pct_rank(sector[c], row[c]), "pct", 0)] for n, c in groups],
                    ["Score", "Z-score", "Universe percentile", "Sector percentile"]), "",
          "Z-scores are sector-neutral and cross-sectional (0 = average stock that month). The composite weights value "
          "and momentum equally by default; quality and earnings quality are shown for information.", ""]

    if risk:
        L += ["## Price risk (36 months, monthly total returns)", "",
              _md_table([["Annualized volatility", _f(risk.get("ann_vol"), "pct")],
                         ["Annualized return", _f(risk.get("cagr"), "pct")],
                         ["Max drawdown", _f(risk.get("max_drawdown"), "pct")],
                         [f"Beta vs {bench}", _f(beta, d=2)]], ["Measure", "Value"]), ""]

    L += ["## Key risks", "",
          "### Analyst assessment" + (f" (retrospective: reviewed {signed:%Y-%m-%d}, after the report date)"
                                      if retro_risks else ""), ""]
    analyst = re.sub(r"<!--.*?-->", "", risks_text, flags=re.S).strip()
    analyst = re.sub(r"^(#+)", r"\1#", analyst, flags=re.M)        # nest under "Analyst assessment"
    L += [analyst, ""] if not draft else [
        f"_Not yet written. Run `python -m eqscreen.research --ticker {t} --init-risks` and fill in "
        f"`config/research/{t}_risks.md`, citing the 10-K risk factors and other sources._", ""]
    L += ["### Mechanical flags from the data", ""]
    L += [f"- {x}" for x in flags] or ["- None triggered."]
    L += ["", "Flag thresholds are this project's choices (config `research.flags`), meant to direct attention, "
              "not to judge.", ""]

    L += ["## Sources", ""]
    recent = versions[versions["period_end"].isin(sorted(versions["period_end"].unique())[-3:])]
    srows = []
    for v in recent.itertuples():
        link = (f"[{v.accession}]({edgar_filing_url(cik, v.accession)})" if cik and isinstance(v.accession, str)
                else (v.accession if isinstance(v.accession, str) else f"{src} snapshot (no accession number)"))
        srows.append([v.period_end, v.available_date + (" (est.)" if v.available_date_is_estimated else ""),
                      v.version_type, link])
    if srows:
        L += [_md_table(srows, ["Fiscal year end", "Public from", "Version", "Filing"], text_cols=(0, 1, 2, 3)), ""]
    prov = "; ".join(f"{s.source} last fetched {s.last} UTC" for s in fetched.itertuples()) or "no fetch log entries"
    man = run.get("manifest", {})
    L += [f"Data retrieval: {prov}. Prices: month-end closes from {price_src}. "
          f"Run manifest: config hash {man.get('config_sha256_12', 'n/a')}, eqscreen "
          f"{man.get('packages', {}).get('eqscreen', 'n/a')}.", ""]

    if assume and scen is not None:
        b = dcf.scenario_assumptions(assume)["base"]
        L += ["## DCF assumptions (base case)", "",
              _md_table([["Revenue growth", ", ".join(_f(x, "pct") for x in b["revenue_growth"])],
                         ["EBIT margin", ", ".join(_f(x, "pct") for x in b["ebit_margin"]) if isinstance(b["ebit_margin"], list)
                          else _f(b["ebit_margin"], "pct")],
                         ["Tax rate", _f(b["tax_rate"], "pct")], ["Terminal growth", _f(b["terminal_growth"], "pct")],
                         ["Risk-free rate", _f(b["risk_free_rate"], "pct", 2)],
                         ["Equity risk premium", _f(b["equity_risk_premium"], "pct", 2)],
                         ["Pre-tax cost of debt", _f(b["pre_tax_cost_of_debt"], "pct", 2)],
                         ["Beta", f"{dcf_res['beta']:.2f}" + (" (set)" if b.get("beta") is not None else " (60m regression)")]],
                        ["Assumption", "Base case"]), "",
              f"Assumptions are the analyst's, recorded in `config/dcf/{t}.yaml` with history for comparison.", ""]
        L += [x.replace("## ", "### ", 1) if x.startswith("## ") else x for x in dcf.render_sources(assume)]

    sv = run.get("survivorship", {})
    L += ["## Model limitations", "",
          "- Annual statements, used only from their publication date; amendments only after they were filed.",
          "- Peer and universe comparisons use the current index list"
          + (f", with history scored only for months a stock was a member. {sv.get('removed_missing_from_data', 'Some')} "
             "companies that left the index are missing from the data (survivorship bias)." if sv.get("membership_data")
             else "; companies that left the index are missing (survivorship bias)."),
          "- DCF: FCFF with Gordon terminal value; no adjustment for leases, minority interests, pensions or options "
          "beyond diluted shares. The result is a valuation under the stated assumptions, not a price target.",
          "- Screen labels are mechanical classifications for prioritizing research, not ratings or recommendations.",
          "- Free data sources can contain errors; check figures against the linked filings before relying on them.", ""]
    return "\n".join(L), {"draft": draft, "dcf": scen is not None, "flags": len(flags), "as_of": m,
                           "retrospective_risks": retro_risks, "retrospective_dcf": retro_dcf,
                           "undated_dcf": undated_dcf}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analyst-ready research report for one company")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--init-risks", action="store_true", help="write a key-risks template to fill in")
    ap.add_argument("--as-of", default=None, help="YYYY-MM (default: latest scored month)")
    ap.add_argument("--assumptions", default=None, help="DCF assumptions file (default config/dcf/<TICKER>.yaml)")
    ap.add_argument("--risks", default=None, help="key risks file (default config/research/<TICKER>_risks.md)")
    ap.add_argument("--mode", choices=["live", "demo"], default="live")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF (needs Playwright + Chromium)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    t = args.ticker.upper()
    risks = Path(args.risks) if args.risks else ROOT / "config" / "research" / f"{t}_risks.md"
    if args.init_risks:
        if risks.exists():
            raise SystemExit(f"{risks} already exists")
        risks.parent.mkdir(parents=True, exist_ok=True)
        risks.write_text(RISKS_TEMPLATE.format(ticker=t), encoding="utf-8")
        print(f"Key-risks template written to {risks}")
        return
    cfg = load_config(args.config, mode=args.mode)
    text, meta = build_report(cfg, t, args.as_of, Path(args.assumptions) if args.assumptions else None, risks)
    out = Path(cfg["project"]["output_dir"]) / "research"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{t}_{meta['as_of']:%Y%m}.md"
    path.write_text(text, encoding="utf-8")
    from .render import html_to_pdf, write_html
    footer = (f"Generated by eqscreen from {cfg['mode']} data as of {meta['as_of']:%Y-%m-%d}. "
              "Not investment advice.")
    html = write_html(path, f"{t} equity research report", footer)
    print(f"Research report written to {path} and {html.name}"
          + (" (DRAFT: key risks not written yet)" if meta["draft"] else ""))
    if args.pdf:
        try:
            print(f"PDF written to {html_to_pdf(html)}")
        except RuntimeError as exc:
            raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
