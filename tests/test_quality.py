import pandas as pd
import pytest

from eqscreen.logging_utils import get_logger
from eqscreen.quality import DataQualityError, persist_and_enforce, run_checks


def _status(checks, name):
    return next(c for c in checks if c.name == name).status


def test_clean_synthetic_data_has_no_failures(fresh_demo_db):
    cfg, conn = fresh_demo_db
    checks = run_checks(cfg, conn)
    assert not [c.name for c in checks if c.status == "fail"]
    assert _status(checks, "share_count_discontinuity") == "pass"


def test_bad_price_fails_and_stops_pipeline(fresh_demo_db):
    cfg, conn = fresh_demo_db
    conn.execute("UPDATE prices_monthly SET close = -1 WHERE ticker = 'SYN001' AND month_end = '2015-06-30'")
    conn.commit()
    checks = run_checks(cfg, conn)
    assert _status(checks, "price_positive") == "fail"
    with pytest.raises(DataQualityError):
        persist_and_enforce(cfg, conn, "test-run", checks, get_logger())
    logged = pd.read_sql_query("SELECT * FROM quality_checks WHERE run_id = 'test-run'", conn)
    assert (logged["check_name"] == "price_positive").any()


def test_missing_split_adjustment_is_flagged(fresh_demo_db):
    cfg, conn = fresh_demo_db
    cfg["fundamentals"]["split_adjust"] = False
    assert _status(run_checks(cfg, conn), "share_count_discontinuity") == "warn"


def test_low_fundamental_coverage_fails(fresh_demo_db):
    cfg, conn = fresh_demo_db
    conn.execute("DELETE FROM fundamentals WHERE ticker > 'SYN020'")
    conn.commit()
    assert _status(run_checks(cfg, conn), "fundamental_coverage") == "fail"


def test_small_universe_is_flagged(fresh_demo_db):
    cfg, conn = fresh_demo_db
    conn.execute("DELETE FROM securities WHERE ticker > 'SYN029'")
    conn.commit()
    assert _status(run_checks(cfg, conn), "universe_breadth") == "warn"


# --------------------------------------------------------------------------- records from an old universe (audit P1)
def _switch_universe(conn, tickers, last_month="2025-12-31"):
    from eqscreen.db import upsert_df

    conn.execute("DELETE FROM securities")
    upsert_df(conn, "securities", pd.DataFrame({"ticker": tickers, "name": "n", "sector": "Industrials",
                                                "sub_industry": None, "cik": None, "updated_at": "x"}))
    months = pd.date_range("2024-01-31", last_month, freq="ME").strftime("%Y-%m-%d")
    upsert_df(conn, "prices_monthly", pd.DataFrame([{"ticker": t, "month_end": m, "close": 10.0, "adj_close": 10.0}
                                                    for t in tickers for m in months]))


def test_coverage_ignores_fundamentals_of_tickers_outside_the_universe(fresh_demo_db):
    cfg, conn = fresh_demo_db
    _switch_universe(conn, [f"NEW{i}" for i in range(25)])            # new universe, no fundamentals
    c = {x.name: x for x in run_checks(cfg, conn)}["fundamental_coverage"]
    assert c.status == "fail" and c.value == 0.0
    assert "0/25 universe tickers" in c.detail and "outside the universe ignored" in c.detail


def test_staleness_ignores_newer_prices_of_excluded_tickers(fresh_demo_db):
    from eqscreen.db import upsert_df

    cfg, conn = fresh_demo_db
    cfg["mode"] = "live"
    _switch_universe(conn, [f"NEW{i}" for i in range(25)], last_month="2025-12-31")
    upsert_df(conn, "prices_monthly", pd.DataFrame([{"ticker": "SYN000", "month_end": "2026-10-31",
                                                     "close": 10.0, "adj_close": 10.0}]))
    c = {x.name: x for x in run_checks(cfg, conn, today=pd.Timestamp("2026-11-12"))}["price_staleness_days"]
    assert c.status == "fail" and c.value == (pd.Timestamp("2026-10-31") - pd.Timestamp("2025-12-31")).days
    assert "latest universe month-end 2025-12-31" in c.detail


def test_one_fresh_ticker_cannot_hide_many_stale_ones(fresh_demo_db):
    from eqscreen.db import upsert_df

    cfg, conn = fresh_demo_db
    tickers = [f"NEW{i}" for i in range(25)]
    _switch_universe(conn, tickers, last_month="2025-06-30")
    upsert_df(conn, "prices_monthly", pd.DataFrame([{"ticker": "NEW0", "month_end": m, "close": 10.0, "adj_close": 10.0}
                                                    for m in ("2025-07-31", "2025-08-31")]))
    c = {x.name: x for x in run_checks(cfg, conn)}["stale_tickers"]
    assert c.status == "warn" and c.value == pytest.approx(24 / 25) and "24/25" in c.detail


# --------------------------------------------------------------------------- integrity scope (audit P3)
def test_bad_record_outside_the_universe_warns_but_does_not_block(fresh_demo_db):
    from eqscreen.db import upsert_df

    cfg, conn = fresh_demo_db
    upsert_df(conn, "prices_monthly", pd.DataFrame([{"ticker": "GONE1", "month_end": "2020-06-15",
                                                     "close": 1.0, "adj_close": 1.0}]))
    checks = {c.name: c for c in run_checks(cfg, conn)}
    assert checks["data_integrity"].status == "pass"
    assert checks["retained_records_integrity"].status == "warn" and "GONE1" in checks["retained_records_integrity"].detail
    assert not [c for c in checks.values() if c.status == "fail"]


def test_bad_record_inside_the_universe_fails_the_run(fresh_demo_db):
    from eqscreen.db import upsert_df

    cfg, conn = fresh_demo_db
    upsert_df(conn, "prices_monthly", pd.DataFrame([{"ticker": "SYN001", "month_end": "2020-06-15",
                                                     "close": 1.0, "adj_close": 1.0}]))
    checks = run_checks(cfg, conn)
    c = {x.name: x for x in checks}["data_integrity"]
    assert c.status == "fail" and "SYN001" in c.detail
    with pytest.raises(DataQualityError, match="data_integrity"):
        persist_and_enforce(cfg, conn, "integrity", checks, get_logger())
