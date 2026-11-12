"""Live data checks against real companies. Skipped unless EQSCREEN_LIVE=1 (needs internet).

    $env:EQSCREEN_LIVE = "1"                 # PowerShell
    $env:EQSCREEN_LIVE_SOURCE = "sec"        # or yfinance / fmp / alphavantage (default: yfinance)
    $env:SEC_USER_AGENT = "Your Name you@example.com"   # required for sec
    python -m pytest tests/test_live.py -v

What it checks, using three well-documented stock splits:
  * the split appears in the price history with the right ratio;
  * implied share count does not jump by the split ratio between the last statement
    filed before the split and the first one after it. A jump means the provider
    already restates history for splits and fundamentals.split_adjust.<source>
    must be set to false (or that adjustment is missing, if the jump is inverted).
"""
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eqscreen.config import ROOT, load_config
from eqscreen.db import connect, table_df
from eqscreen.factors import compute_ratios, load_factor_inputs
from eqscreen.pipeline import run

pytestmark = pytest.mark.skipif(os.environ.get("EQSCREEN_LIVE") != "1",
                                reason="live data check; set EQSCREEN_LIVE=1 to run")

# Effective dates of real splits as listed by the companies / exchanges.
SPLITS = {"AAPL": ("2020-08-31", 4.0), "TSLA": ("2022-08-25", 3.0), "NVDA": ("2024-06-10", 10.0)}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live")
    source = os.environ.get("EQSCREEN_LIVE_SOURCE", "yfinance")
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    universe = tmp / "universe.csv"
    pd.DataFrame({"ticker": list(SPLITS), "sector": ["Information Technology", "Consumer Discretionary",
                                                     "Information Technology"]}).to_csv(universe, index=False)
    cfg["project"].update(db_path=str(tmp / "live.db"), output_dir=str(tmp / "out"),
                          raw_dir=str(tmp / "raw"), log_dir=str(tmp / "logs"))
    cfg["universe"].update(source="file", file=str(universe))
    cfg["prices"]["start"] = "2018-01-01"
    cfg["fundamentals"]["source"] = source
    if source == "sec":
        cfg["fundamentals"]["sec_user_agent"] = os.environ["SEC_USER_AGENT"]
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run(mode="live", stages=["universe", "prices", "fundamentals"], config_path=str(path))
    c = load_config(path, mode="live")
    conn = connect(c["project"]["db_path"])
    splits = table_df(conn, "SELECT * FROM splits", date_cols=("split_date",))
    ratios = compute_ratios(load_factor_inputs(conn, c), splits, c)
    yield c, conn, splits, ratios
    conn.close()


@pytest.mark.parametrize("ticker", list(SPLITS))
def test_split_is_in_price_history(live, ticker):
    _, _, splits, _ = live
    date, ratio = SPLITS[ticker]
    s = splits[(splits["ticker"] == ticker)
               & ((splits["split_date"] - pd.Timestamp(date)).abs() <= pd.Timedelta(days=3))]
    assert len(s) == 1, f"{ticker}: split near {date} missing from price history"
    assert s.iloc[0]["ratio"] == pytest.approx(ratio, rel=0.01)


@pytest.mark.parametrize("ticker", list(SPLITS))
def test_share_count_continuous_across_split(live, ticker):
    cfg, _, _, ratios = live
    date, ratio = SPLITS[ticker]
    stmts = (ratios[ratios["ticker"] == ticker].dropna(subset=["shares_adj", "available_date"])
             .drop_duplicates(["period_end", "available_date"]).sort_values("available_date"))
    before = stmts[stmts["available_date"] < date]
    after = stmts[stmts["available_date"] >= date]
    if before.empty or after.empty:
        pytest.skip(f"{ticker}: {cfg['fundamentals']['source']} history does not span the {date} split")
    jump = after.iloc[0]["shares_adj"] / before.iloc[-1]["shares_adj"]
    assert 0.8 <= jump <= 1.25, (
        f"{ticker}: split-adjusted share count changes {jump:.2f}x across the {date} {ratio:g}-for-1 split. "
        + (f"The provider already restates for splits: set fundamentals.split_adjust.{cfg['fundamentals']['source']}: false."
           if jump < 1 / 1.25 else "Split adjustment appears to be missing.")
    )


def test_latest_valuation_is_usable(live):
    _, _, _, ratios = live
    latest = ratios.sort_values("month_end").groupby("ticker").tail(1)
    assert (latest["market_cap"] > 0).all()
    assert latest["ev"].notna().all()
