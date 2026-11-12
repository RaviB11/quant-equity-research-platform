"""Real-market validation report. Run after a live pipeline run:

    python -m eqscreen.validate               # writes outputs/validation/validation_report.md
    python -m eqscreen.validate --archive     # also preserves the inputs as data/archive/inputs_<run>.zip

Sections: data provenance; filing-availability validation (filing lags, revisions, accession
coverage, a point-in-time check); corporate actions; data-quality results; survivorship
disclosure; and real results side by side with the synthetic benchmark, so differences are
documented rather than assumed. The synthetic demo is generated (in its own database) if it
has not been run yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .db import connect, table_df
from .factors import LookaheadError, load_factor_inputs
from .note import _f, _md_table
from .quality import _share_discontinuity
from .universe import survivorship_summary


# --------------------------------------------------------------------------- diagnostics
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def diagnostics(cfg: dict) -> dict:
    """Headline numbers from one mode's outputs and database, for side-by-side comparison."""
    out = Path(cfg["project"]["output_dir"])
    d: dict = {}
    run = json.loads((out / "run_summary.json").read_text()) if (out / "run_summary.json").exists() else {}
    d["universe"] = run.get("n_universe")
    s = _read(out / "backtest_summary.csv")
    head_h = int(cfg["backtest"]["holding_months"][0])

    def srow(strategy, series="strategy_net"):
        if s.empty:
            return None
        r = s[(s["strategy"] == strategy) & (s["series"] == series)]
        if strategy == "composite":
            r = r[r["holding"] == head_h]
        return r.iloc[0] if len(r) else None
    net, ew = srow("composite"), srow("composite", "ew_universe")
    if net is not None:
        d.update({"period": f"{net['start']} to {net['end']}", "cagr": net["cagr"], "ew_cagr": ew["cagr"],
                  "sharpe": net["sharpe"], "excess": net["excess_return_ann"], "excess_t": net["t_stat_excess"],
                  "max_dd": net["max_drawdown"], "holdings": net["avg_holdings"],
                  "missing_share": net.get("missing_return_share")})
    for name, key in (("lag0", "composite_lag0"), ("missing_loss", "composite_missing_loss")):
        r = srow(key)
        d[f"{name}_cagr"] = r["cagr"] if r is not None else np.nan
    ic = _read(out / "ic_summary.csv")
    for score, h in (("value_score", 1), ("value_score", 12), ("momentum_score", 1), ("composite", 1),
                     ("earnings_quality_score", 1)):
        r = ic[(ic.get("score") == score) & (ic.get("horizon_months") == h)] if len(ic) else ic
        d[f"ic_{score}_{h}"] = (r["mean_ic"].iloc[0], r["t_stat"].iloc[0]) if len(r) else (np.nan, np.nan)
    ev = _read(out / "ev_coverage.csv")
    d["ev_coverage"] = ev["ev_input_coverage"].iloc[-1] if len(ev) else np.nan
    conn = connect(cfg["project"]["db_path"])
    src = cfg["fundamentals"]["source"]
    f = table_df(conn, "SELECT f.ticker, f.period_end, f.available_date_is_estimated FROM fundamentals f "
                       "JOIN securities s ON s.ticker = f.ticker WHERE f.source = ?", (src,))
    d["estimated_share"] = f["available_date_is_estimated"].mean() if len(f) else np.nan
    d["revised_share"] = (f.groupby(["ticker", "period_end"]).size() > 1).mean() if len(f) else np.nan
    d["survivorship"] = survivorship_summary(conn)
    conn.close()
    return d


def _ic(pair) -> str:
    ic, t = pair
    return "n/a" if pd.isna(ic) else f"{ic:.3f} (t {t:.1f})"


def comparison_rows(real: dict, synth: dict) -> list[list[str]]:
    def p(x):
        return _f(x, "pct")

    rows = [
        ["Universe size", str(synth.get("universe")), str(real.get("universe")),
         "Synthetic size is fixed in config; real is the index list."],
        ["Backtest period", str(synth.get("period")), str(real.get("period")),
         "Real history is limited by how far back statements reach (source-dependent)."],
        ["Composite CAGR, net", p(synth.get("cagr")), p(real.get("cagr")), ""],
        ["Equal-weight universe CAGR", p(synth.get("ew_cagr")), p(real.get("ew_cagr")), ""],
        ["Excess return a year (Newey-West t)", f"{p(synth.get('excess'))} ({_f(synth.get('excess_t'), d=1)})",
         f"{p(real.get('excess'))} ({_f(real.get('excess_t'), d=1)})",
         "Synthetic excess comes from a planted value effect; real excess, if any, is what the market delivered."],
        ["Sharpe ratio", _f(synth.get("sharpe"), d=2), _f(real.get("sharpe"), d=2), ""],
        ["Max drawdown", p(synth.get("max_dd")), p(real.get("max_dd")), ""],
        ["Same-close CAGR (optimistic)", p(synth.get("lag0_cagr")), p(real.get("lag0_cagr")),
         "The gap to the headline shows how much execution timing matters."],
        ["Missing-return share of holdings", p(synth.get("missing_share")), p(real.get("missing_share")),
         "Synthetic data have no gaps by construction."],
        ["CAGR if missing returns were total losses", p(synth.get("missing_loss_cagr")), p(real.get("missing_loss_cagr")), ""],
        ["Value IC, 1 month", _ic(synth.get("ic_value_score_1", (np.nan,) * 2)), _ic(real.get("ic_value_score_1", (np.nan,) * 2)),
         "Planted in synthetic data, so a strong synthetic IC is expected."],
        ["Value IC, 12 months", _ic(synth.get("ic_value_score_12", (np.nan,) * 2)),
         _ic(real.get("ic_value_score_12", (np.nan,) * 2)), ""],
        ["Momentum IC, 1 month", _ic(synth.get("ic_momentum_score_1", (np.nan,) * 2)),
         _ic(real.get("ic_momentum_score_1", (np.nan,) * 2)), "Not planted in synthetic data (should be near zero there)."],
        ["Earnings-quality IC, 1 month", _ic(synth.get("ic_earnings_quality_score_1", (np.nan,) * 2)),
         _ic(real.get("ic_earnings_quality_score_1", (np.nan,) * 2)), "Not planted in synthetic data."],
        ["Statements with estimated filing dates", p(synth.get("estimated_share")), p(real.get("estimated_share")),
         "Synthetic data have exact filing dates; yfinance and Alpha Vantage have none."],
        ["Fiscal periods with revisions", p(synth.get("revised_share")), p(real.get("revised_share")),
         "Synthetic data have no restatements."],
        ["EV input coverage (latest month)", p(synth.get("ev_coverage")), p(real.get("ev_coverage")), ""],
    ]
    return rows


def observations(real: dict) -> list[str]:
    """Mechanical statements about the real results, based only on the computed numbers."""
    obs = []
    for label, key in (("Value", "ic_value_score_1"), ("Momentum", "ic_momentum_score_1"),
                       ("Composite", "ic_composite_1"), ("Earnings quality", "ic_earnings_quality_score_1")):
        ic, t = real.get(key, (np.nan, np.nan))
        if pd.isna(ic):
            continue
        if abs(t) >= 2:
            obs.append(f"{label}: mean 1-month rank IC {ic:.3f} with Newey-West t {t:.1f}, statistically distinguishable "
                       f"from zero ({'positive' if ic > 0 else 'negative'}).")
        else:
            obs.append(f"{label}: mean 1-month rank IC {ic:.3f} (t {t:.1f}); not distinguishable from zero over this sample.")
    if pd.notna(real.get("excess_t")):
        obs.append(f"Composite excess return over the equal-weight universe: {_f(real['excess'], 'pct')} a year, "
                   f"Newey-West t {real['excess_t']:.1f}"
                   + (" (below 2: weak evidence)." if abs(real["excess_t"]) < 2 else "."))
    if pd.notna(real.get("lag0_cagr")) and pd.notna(real.get("cagr")):
        obs.append(f"Same-close trading would have reported {_f(real['lag0_cagr'], 'pct')} instead of "
                   f"{_f(real['cagr'], 'pct')} a year.")
    return obs


# --------------------------------------------------------------------------- filing validation
def filing_validation(cfg: dict) -> tuple[list[str], pd.DataFrame]:
    conn = connect(cfg["project"]["db_path"])
    src = cfg["fundamentals"]["source"]
    # Current universe only: records left over from earlier universes are not evidence about this one.
    f = table_df(conn, "SELECT f.* FROM fundamentals f JOIN securities s ON s.ticker = f.ticker WHERE f.source = ?",
                 (src,), date_cols=("period_end", "available_date"))
    splits = table_df(conn, "SELECT sp.* FROM splits sp JOIN securities s ON s.ticker = sp.ticker",
                      date_cols=("split_date",))
    qc = table_df(conn, "SELECT * FROM quality_checks WHERE run_id = (SELECT run_id FROM quality_checks "
                        "ORDER BY id DESC LIMIT 1)")
    fl = table_df(conn, "SELECT source, status, COUNT(*) AS n FROM fetch_log GROUP BY source, status ORDER BY source, status")
    try:
        inputs = load_factor_inputs(conn, cfg)            # asserts no statement is used before it was public
        pit_ok, used = True, inputs["available_date"].notna().sum()
    except LookaheadError:
        pit_ok, used = False, 0
    conn.close()
    L = []
    if f.empty:
        return ["No fundamentals stored for this source."], pd.DataFrame()
    orig = f[f["version_type"] == "original"].copy()
    orig["lag_days"] = (orig["available_date"] - orig["period_end"]).dt.days
    real_dates = orig[orig["available_date_is_estimated"] == 0]
    q = real_dates["lag_days"].quantile([0.1, 0.5, 0.9]) if len(real_dates) else pd.Series(dtype=float)
    kinds = f["version_type"].value_counts().to_dict()
    per = f.groupby(["ticker", "period_end"]).size()
    L += [_md_table([
        ["Statements stored (all versions)", f"{len(f):,}"],
        ["Companies / fiscal periods", f"{f['ticker'].nunique():,} / {len(per):,}"],
        ["Originals with a real filing date", _f(1 - orig["available_date_is_estimated"].mean(), "pct")],
        ["Filing lag after fiscal year-end, 10th / 50th / 90th percentile (days)",
         " / ".join(f"{v:.0f}" for v in q.values) if len(q) else "n/a (no real filing dates)"],
        ["Originals filed more than 120 days after year-end", f"{int((real_dates['lag_days'] > 120).sum()):,}"],
        ["Fiscal periods with more than one version", f"{int((per > 1).sum()):,} ({(per > 1).mean():.1%})"],
        ["Versions by type", ", ".join(f"{k}: {v:,}" for k, v in kinds.items())],
        ["Versions with an SEC accession number", _f(f["accession"].notna().mean(), "pct")],
        ["Point-in-time check (no statement used before its public date)",
         f"passed on {used:,} stock-months" if pit_ok else "FAILED"],
    ], ["Check", "Result"]), ""]
    late = real_dates.nlargest(10, "lag_days")[["ticker", "period_end", "available_date", "lag_days"]]
    if len(late):
        L += ["Longest filing lags among originals (worth spot-checking against EDGAR):", "",
              _md_table([[r.ticker, r.period_end.strftime("%Y-%m-%d"), r.available_date.strftime("%Y-%m-%d"),
                          str(r.lag_days)] for r in late.itertuples()],
                        ["Ticker", "Fiscal year end", "Public from", "Days"], text_cols=(0, 1, 2)), ""]
    disc = _share_discontinuity(f, splits, True) if len(f) else None
    L += ["## Corporate actions", "",
          f"{len(splits):,} splits recorded across {splits['ticker'].nunique() if len(splits) else 0} tickers. "
          + (f"Share-count discontinuity check: {disc.status} ({disc.detail})." if disc else ""), "",
          "Run `tests/test_live.py` (EQSCREEN_LIVE=1) and `tests/test_real_fixture.py` for the AAPL / TSLA / NVDA split "
          "checks against real data.", ""]
    if len(qc):
        L += ["## Data quality checks (latest run)", "",
              _md_table([[r.check_name, r.status, str(r.detail)] for r in qc.itertuples()],
                        ["Check", "Status", "Detail"], text_cols=(0, 1, 2)), ""]
    if len(fl):
        L += ["## Fetch log", "", _md_table([[r.source, r.status, f"{r.n:,}"] for r in fl.itertuples()],
                                              ["Source", "Status", "Count"], text_cols=(0, 1)), ""]
    lags = orig[["ticker", "period_end", "available_date", "available_date_is_estimated", "lag_days"]]
    return L, lags


# --------------------------------------------------------------------------- archive
def archive_inputs(cfg: dict) -> Path:
    """Zip the database, configuration and run metadata so a result can be re-examined later.

    Stays on your machine (data/ is git-ignored): provider terms generally forbid
    redistributing the downloaded data.
    """
    out = Path(cfg["project"]["output_dir"])
    run = json.loads((out / "run_summary.json").read_text()) if (out / "run_summary.json").exists() else {}
    db = Path(cfg["project"]["db_path"])
    dest = db.parent / "archive" / f"inputs_{run.get('run_id', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    raw = Path(cfg["project"]["raw_dir"])
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(db, db.name)
        z.write(ROOT / "config" / "config.yaml", "config.yaml")
        for extra in ("run_summary.json", "quality_checks.csv", "ev_coverage.csv"):
            if (out / extra).exists():
                z.write(out / extra, extra)
        for csv in sorted(raw.glob("universe_sp500_*.csv"))[-1:] + sorted(raw.glob("sp500_changes_*.csv"))[-1:]:
            z.write(csv, f"data_raw/{csv.name}")
        for yml in (ROOT / "config" / "dcf").glob("*.yaml"):
            z.write(yml, f"config/dcf/{yml.name}")
        z.writestr("ARCHIVE_MANIFEST.json", json.dumps({
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run_id": run.get("run_id"),
            "database_sha256": digest, "manifest": run.get("manifest")}, indent=2))
    return dest


# --------------------------------------------------------------------------- report
def build(cfg: dict, synth_cfg: dict, archive: bool = False) -> Path:
    out = Path(cfg["project"]["output_dir"]) / "validation"
    out.mkdir(parents=True, exist_ok=True)
    real, synth = diagnostics(cfg), diagnostics(synth_cfg)
    rows = comparison_rows(real, synth)
    pd.DataFrame(rows, columns=["metric", "synthetic", "real", "note"]).to_csv(out / "real_vs_synthetic.csv", index=False)
    filing_lines, lags = filing_validation(cfg)
    lags.to_csv(out / "filing_lags.csv", index=False)
    sv = real["survivorship"]
    is_demo = cfg["mode"] == "demo"
    L = []
    if is_demo:
        L += ["> **Demo mode:** both columns below are synthetic. Run `python -m eqscreen.validate` after a live run.", ""]
    L += ["# Real-market validation", "",
          f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the {cfg['mode']} database "
          f"(fundamentals: {cfg['fundamentals']['source']}).", "",
          "## Filing availability", ""] + filing_lines
    L += ["## Survivorship", ""]
    if sv.get("membership_data"):
        L += [f"Index membership was rebuilt from the dated change history, recorded back to "
              f"{sv['earliest_recorded_change']}. Stocks are scored only while they were members. "
              f"{sv['removed_since_first_price']} companies left the index after {sv['first_price_month']}; "
              f"{sv['removed_with_price_data']} of them have price data and {sv['removed_missing_from_data']} do not. "
              + ("The missing departures (often poor performers) are absent from the backtest, so results are "
                 "biased upward. " if sv["removed_missing_from_data"] else "")
              + "Membership before the earliest recorded change is assumed, not observed.", ""]
    else:
        L += ["No index-change history is stored (file universe or parse failure), so the backtest applies today's "
              "members to the past: survivorship bias in full.", ""]
    L += ["A complete fix needs point-in-time constituent lists with delisting returns (for example from a "
          "commercial survivorship-free database).", "",
          "## Real results against the synthetic benchmark", "",
          "The synthetic market has a planted value effect and no momentum or earnings-quality effect, so it shows "
          "what the pipeline reports when the answer is known. Differences below are expected; the point is to "
          "document them, not to make the real results look like the synthetic ones.", "",
          _md_table(rows, ["Metric", "Synthetic", "Real", "Why they can differ"], text_cols=(0, 3)), "",
          "### Observations on the real results", ""]
    L += [f"- {o}" for o in observations(real)] or ["- No real backtest outputs found."]
    if archive:
        dest = archive_inputs(cfg)
        L += ["", "## Preserved inputs", "",
              f"Inputs archived to `{dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest}` (database, "
              "configuration, DCF assumptions, universe and change lists, run manifest, database SHA-256). It stays "
              "local: provider terms generally forbid redistributing the data.", ""]
    path = out / "validation_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    from .render import write_html
    write_html(path, "Real-market validation", f"Generated by eqscreen from the {cfg['mode']} database.")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Real-market validation report")
    ap.add_argument("--mode", choices=["live", "demo"], default="live")
    ap.add_argument("--archive", action="store_true", help="also zip the inputs into data/archive/")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, mode=args.mode)
    synth_cfg = load_config(args.config, mode="demo")
    if not (Path(synth_cfg["project"]["output_dir"]) / "backtest_summary.csv").exists():
        from .pipeline import run
        print("Synthetic benchmark not found; running the demo pipeline first ...")
        run(mode="demo", config_path=args.config)
    print(f"Validation report written to {build(cfg, synth_cfg, archive=args.archive)}")


if __name__ == "__main__":
    main()
