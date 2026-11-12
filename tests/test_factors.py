import numpy as np
import pandas as pd
import pytest

from eqscreen.factors import build_factor_panel, compute_ratios, load_factor_inputs, split_factor
from eqscreen.prices import daily_to_monthly, last_complete_month_end


# --------------------------------------------------------------------------- prices
def test_last_complete_month_end():
    assert last_complete_month_end(pd.Timestamp("2026-11-12")) == pd.Timestamp("2026-10-31")
    assert last_complete_month_end(pd.Timestamp("2026-01-01")) == pd.Timestamp("2025-12-31")


def test_daily_to_monthly_drops_partial_month_and_reads_splits():
    idx = pd.bdate_range("2024-01-02", "2024-03-15")
    n = len(idx)
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "Stock Splits"], ["AAA", "BBB"]],
                                      names=["Price", "Ticker"])
    data = pd.DataFrame(0.0, index=idx, columns=cols)
    data[("Close", "AAA")] = np.arange(n) + 100.0
    data[("Adj Close", "AAA")] = np.arange(n) + 99.0
    data[("Close", "BBB")] = 50.0
    data[("Adj Close", "BBB")] = 49.0
    data.loc[pd.Timestamp("2024-02-12"), ("Stock Splits", "AAA")] = 4.0
    monthly, splits = daily_to_monthly(data, ["AAA", "BBB"], include_partial_month=False,
                                       today=pd.Timestamp("2024-03-15"))
    assert sorted(monthly["month_end"].unique()) == ["2024-01-31", "2024-02-29"]
    jan = monthly[(monthly["ticker"] == "AAA") & (monthly["month_end"] == "2024-01-31")].iloc[0]
    assert jan["close"] == 100.0 + list(idx).index(pd.Timestamp("2024-01-31"))
    assert splits.to_dict("records") == [{"ticker": "AAA", "split_date": "2024-02-12", "ratio": 4.0}]


# --------------------------------------------------------------------------- split adjustment
def test_split_factor_counts_only_later_splits():
    splits = pd.DataFrame({"ticker": ["AAA", "AAA"], "split_date": pd.to_datetime(["2020-06-01", "2022-01-03"]),
                           "ratio": [4.0, 2.0]})
    df = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "AAA", "BBB"],
                       "d": pd.to_datetime(["2019-12-31", "2021-01-01", "2022-06-30", "2020-06-01", "2019-12-31"])})
    assert list(split_factor(df, splits, "d")) == [8.0, 2.0, 1.0, 2.0, 1.0]


def _ratio_row(**kw):
    base = dict(ticker="AAA", sector="Industrials", close=50.0, shares_diluted=100.0, eps_diluted=2.0,
                net_income=200.0, total_debt=1000.0, cash=500.0, total_equity=2500.0, ebit=400.0, ebitda=500.0,
                available_date=pd.Timestamp("2023-03-01"), available_date_prev=pd.Timestamp("2022-03-01"),
                period_end=pd.Timestamp("2022-12-31"), period_end_prev=pd.Timestamp("2021-12-31"),
                eps_prev=1.6, net_income_prev=160.0, shares_prev=100.0,
                operating_cash_flow=260.0, capex=60.0, total_assets=4200.0, total_assets_prev=3800.0, revenue=2000.0)
    base.update(kw)
    return base


CFG = {"fundamentals": {"split_adjust": True}, "factors": {"ev_excluded_sectors": ["Financials"]}}


def test_compute_ratios_values_and_edge_cases():
    rows = pd.DataFrame([
        _ratio_row(),
        _ratio_row(ticker="NEG", eps_diluted=-1.0, net_income=-100.0),
        _ratio_row(ticker="BNK", sector="Financials"),
    ])
    out = compute_ratios(rows, pd.DataFrame(columns=["ticker", "split_date", "ratio"]), CFG).set_index("ticker")
    a = out.loc["AAA"]
    assert a["market_cap"] == 5000 and a["ev"] == 5500
    assert a["pe"] == 25 and a["pb"] == 2 and a["ev_ebit"] == pytest.approx(13.75)
    assert a["eps_growth"] == pytest.approx(0.25) and a["peg"] == pytest.approx(1.0)
    assert np.isnan(out.loc["NEG", "pe"]) and out.loc["NEG", "earnings_yield"] == pytest.approx(-0.02)
    assert np.isnan(out.loc["BNK", "ev_ebit"]) and out.loc["BNK", "pe"] == 25


def test_compute_ratios_applies_later_split():
    splits = pd.DataFrame({"ticker": ["AAA"], "split_date": pd.to_datetime(["2024-01-15"]), "ratio": [2.0]})
    out = compute_ratios(pd.DataFrame([_ratio_row()]), splits, CFG).iloc[0]
    # Reported 100 shares pre-split = 200 post-split shares at the split-adjusted price.
    assert out["shares_adj"] == 200 and out["market_cap"] == 10000 and out["pe"] == 50


# --------------------------------------------------------------------------- point in time
def test_factor_inputs_are_point_in_time(demo_run, demo_conn):
    df = load_factor_inputs(demo_conn, demo_run)
    used = df.dropna(subset=["available_date"])
    assert len(used) > 1000
    assert (used["available_date"] <= used["month_end"]).all()
    stale = (used["month_end"] - used["available_date"]).dt.days
    assert stale.max() <= demo_run["fundamentals"]["max_staleness_months"] * 31


def test_no_lookahead_scores_unchanged_when_future_removed(fresh_demo_db):
    """Scores at or before date X must not change if everything after X is deleted."""
    cfg, conn = fresh_demo_db
    full = build_factor_panel(conn, cfg)
    cutoff = "2017-06-30"
    conn.execute("DELETE FROM prices_monthly WHERE month_end > ?", (cutoff,))
    conn.execute("DELETE FROM fundamentals WHERE available_date > ?", (cutoff,))
    conn.commit()
    trunc = build_factor_panel(conn, cfg)
    cols = ["composite", "value_score", "momentum_score", "z_earnings_yield", "z_momentum_12_1", "pe", "label"]
    a = full[full["month_end"] <= cutoff].set_index(["month_end", "ticker"])[cols].sort_index()
    b = trunc.set_index(["month_end", "ticker"])[cols].sort_index()
    assert a["composite"].notna().sum() > 1000
    pd.testing.assert_frame_equal(a, b.loc[a.index])


# --------------------------------------------------------------------------- restatements
def _pick_ticker(conn):
    f = pd.read_sql_query("SELECT ticker, period_end, available_date FROM fundamentals", conn)
    has = f.pivot_table(index="ticker", columns="period_end", values="available_date", aggfunc="first")
    ok = has[has["2017-12-31"].notna() & has["2018-12-31"].notna()]
    return ok.index[0], ok.loc[ok.index[0], "2018-12-31"]


def test_amended_filing_only_affects_scores_after_its_filing_date(fresh_demo_db):
    """Reviewer's test: an amendment with a very different EPS must not leak backwards."""
    cfg, conn = fresh_demo_db
    tk, fy18_filed = _pick_ticker(conn)
    base = build_factor_panel(conn, cfg)
    orig = pd.read_sql_query("SELECT * FROM fundamentals WHERE ticker = ? AND period_end = '2017-12-31'", conn, params=(tk,))
    amended = orig.assign(available_date="2018-09-14", version_type="amendment",
                          eps_diluted=orig["eps_diluted"] * 5, net_income=orig["net_income"] * 5)
    from eqscreen.db import upsert_df
    upsert_df(conn, "fundamentals", amended)
    after = build_factor_panel(conn, cfg)

    key, cols = ["month_end", "ticker"], ["composite", "value_score", "pe", "earnings_yield", "label"]
    before_cut = lambda d: d[d["month_end"] <= "2018-08-31"].set_index(key)[cols].sort_index()  # noqa: E731
    pd.testing.assert_frame_equal(before_cut(base), before_cut(after))

    window = (after["ticker"] == tk) & after["month_end"].between("2018-09-30", "2018-12-31")
    b = base[(base["ticker"] == tk) & base["month_end"].between("2018-09-30", "2018-12-31")]
    assert (after.loc[window, "version_type"] == "amendment").all()
    np.testing.assert_allclose(after.loc[window, "earnings_yield"].to_numpy(), b["earnings_yield"].to_numpy() * 5)

    # Once FY2018 is out, the amended FY2017 EPS feeds growth as the prior year.
    later = (after["ticker"] == tk) & (after["month_end"] == pd.Timestamp(fy18_filed) + pd.offsets.MonthEnd(1))
    later_base = (base["ticker"] == tk) & (base["month_end"] == pd.Timestamp(fy18_filed) + pd.offsets.MonthEnd(1))
    assert after.loc[later, "eps_growth"].iloc[0] != base.loc[later_base, "eps_growth"].iloc[0]


def test_provider_snapshot_change_becomes_observed_revision(fresh_demo_db):
    from eqscreen.db import upsert_df
    from eqscreen.fundamentals import FUND_COLS, version_against_store

    cfg, conn = fresh_demo_db
    tk, _ = _pick_ticker(conn)
    snap = pd.read_sql_query("SELECT * FROM fundamentals WHERE ticker = ?", conn, params=(tk,))[FUND_COLS]
    changed = snap["period_end"] == "2018-12-31"
    snap.loc[changed, "net_income"] *= 0.5
    out = version_against_store(conn, snap, "synthetic", today=pd.Timestamp("2020-01-15"))
    assert len(out) == 1 and out.iloc[0]["period_end"] == "2018-12-31"
    assert out.iloc[0]["version_type"] == "revision_observed" and out.iloc[0]["available_date"] == "2020-01-15"
    upsert_df(conn, "fundamentals", out)
    inputs = load_factor_inputs(conn, cfg)
    row = lambda m: inputs[(inputs["ticker"] == tk) & (inputs["month_end"] == m)].iloc[0]  # noqa: E731
    assert row("2019-12-31")["version_type"] == "original"
    assert row("2020-01-31")["version_type"] == "revision_observed"
    assert row("2020-01-31")["net_income"] == pytest.approx(row("2019-12-31")["net_income"] * 0.5)


def test_missing_debt_or_cash_blanks_ev_unless_assume_zero():
    rows = pd.DataFrame([_ratio_row(ticker="NODEBT", total_debt=np.nan), _ratio_row(ticker="NOCASH", cash=np.nan)])
    empty = pd.DataFrame(columns=["ticker", "split_date", "ratio"])
    out = compute_ratios(rows, empty, CFG).set_index("ticker")
    assert out[["ev", "ev_ebit", "ebit_ev", "ebitda_ev"]].isna().all().all()
    assert (out["ev_inputs_complete"] == 0).all()
    assert out.loc["NODEBT", "pe"] == 25                     # non-EV ratios unaffected
    lenient = {**CFG, "factors": {**CFG["factors"], "ev_missing_policy": "assume_zero"}}
    out2 = compute_ratios(rows, empty, lenient).set_index("ticker")
    assert out2.loc["NODEBT", "ev"] == 5000 - 500 and out2.loc["NOCASH", "ev"] == 5000 + 1000


def test_ev_coverage_report_counts_missing_inputs(fresh_demo_db):
    from eqscreen.factors import ev_coverage

    cfg, conn = fresh_demo_db
    conn.execute("UPDATE fundamentals SET total_debt = NULL WHERE ticker IN ('SYN001', 'SYN002')")
    conn.commit()
    panel = build_factor_panel(conn, cfg)
    cov = ev_coverage(panel, cfg)
    last = cov.iloc[-1]
    excluded = panel[(panel["month_end"] == last["month_end"]) & panel["ticker"].isin(["SYN001", "SYN002"])
                     & panel["period_end"].notna()]
    n_missing = int((excluded["sector"] != "Financials").sum())
    assert last["missing_debt"] == n_missing
    assert last["ev_inputs_complete"] == last["with_fundamentals"] - n_missing
    assert excluded.loc[excluded["sector"] != "Financials", "ebit_ev"].isna().all()


def test_earnings_quality_metrics():
    out = compute_ratios(pd.DataFrame([_ratio_row(), _ratio_row(ticker="LOSS", net_income=-50.0, eps_diluted=-0.5),
                                       _ratio_row(ticker="NOPREV", total_assets_prev=np.nan)]),
                         pd.DataFrame(columns=["ticker", "split_date", "ratio"]), CFG).set_index("ticker")
    a = out.loc["AAA"]
    assert a["fcf"] == 200
    assert a["accruals_ratio"] == pytest.approx((200 - 260) / 4000)     # NI below OCF: negative accruals
    assert a["accruals_low"] == pytest.approx(-a["accruals_ratio"])
    assert a["cash_conversion"] == pytest.approx(1.3)
    assert a["fcf_yield"] == pytest.approx(200 / 5000) and a["fcf_margin"] == pytest.approx(0.1)
    assert np.isnan(out.loc["LOSS", "cash_conversion"])                # undefined for losses
    assert out.loc["LOSS", "accruals_ratio"] == pytest.approx((-50 - 260) / 4000)
    assert np.isnan(out.loc["NOPREV", "accruals_ratio"])               # needs two balance sheets


# --------------------------------------------------------------------------- guarantees survive python -O
def test_point_in_time_guard_raises_explicitly():
    from eqscreen.factors import LookaheadError, check_point_in_time

    df = pd.DataFrame({"month_end": pd.to_datetime(["2020-01-31"]), "available_date": pd.to_datetime(["2020-02-15"]),
                       "available_date_prev": pd.to_datetime(["2019-02-15"])})
    with pytest.raises(LookaheadError, match="1 stock-months"):
        check_point_in_time(df)


def test_guards_still_fire_under_python_optimize_flag():
    import subprocess
    import sys

    code = ("import pandas as pd\nfrom eqscreen.factors import LookaheadError, check_point_in_time\n"
            "df = pd.DataFrame({'month_end': pd.to_datetime(['2020-01-31']), "
            "'available_date': pd.to_datetime(['2020-02-15'])})\n"
            "try:\n    check_point_in_time(df)\nexcept LookaheadError:\n    print('raised')\n")
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "raised"


def test_no_assert_statements_guard_correctness_in_src():
    """asserts are stripped by python -O; every guarantee in src must raise explicitly."""
    import ast
    from eqscreen.config import ROOT

    offenders = [f"{p.name}:{n.lineno}" for p in (ROOT / "src" / "eqscreen").glob("*.py")
                 for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) if isinstance(n, ast.Assert)]
    assert offenders == []
