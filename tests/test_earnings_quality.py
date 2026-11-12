import numpy as np
import pandas as pd
import pytest

from eqscreen import earnings_quality as eq
from eqscreen.db import upsert_df


def _hist(fcf, cc, acc=None):
    n = len(fcf)
    return pd.DataFrame({"period_end": pd.date_range("2018-12-31", periods=n, freq="YE"), "fcf": fcf,
                         "cash_conversion": cc, "accruals_ratio": acc if acc is not None else [0.0] * n})


CFG = {"research": {"flags": {}}}


def test_flags_fire_on_weak_cash_and_falling_fcf():
    s = eq.summarize(_hist([100, 90, 80, 70], [1.1, 0.7, 0.6, 0.9]), None, "T", CFG)
    text = " ".join(s["flags"])
    assert "2 of the last 3 years" in text and "fell in each of the last 3 years" in text
    assert s["fcf_cagr"] == pytest.approx((70 / 100) ** (1 / 3) - 1, rel=1e-3)


def test_no_flags_for_steady_cash_generator():
    s = eq.summarize(_hist([100, 110, 120, 130], [1.2, 1.1, 1.3, 1.2]), None, "T", CFG)
    assert s["flags"] == [] and s["fcf_positive_years"] == 4


def test_negative_fcf_and_no_cagr_through_zero():
    s = eq.summarize(_hist([-20, 30, -10], [0.5, 1.2, 1.0]), None, "T", CFG)
    assert "negative in the latest" in " ".join(s["flags"]) and "fcf_cagr" not in s


def test_accruals_percentile_uses_the_cross_section():
    cross = pd.DataFrame({"ticker": [f"S{i}" for i in range(30)] + ["T"],
                          "accruals_ratio": list(np.linspace(-0.1, 0.05, 30)) + [0.2]})
    s = eq.summarize(_hist([100, 110], [1.1, 1.2], [0.1, 0.2]), cross, "T", CFG)
    assert s["accruals_percentile"] == 1.0 and "highest 20%" in " ".join(s["flags"])


def test_history_is_point_in_time(fresh_demo_db):
    cfg, conn = fresh_demo_db
    row = pd.read_sql_query("SELECT * FROM fundamentals WHERE ticker = 'SYN003' AND period_end = '2016-12-31'", conn)
    upsert_df(conn, "fundamentals", row.assign(available_date="2018-06-15", version_type="amendment",
                                               operating_cash_flow=row["operating_cash_flow"] * 3))
    before = eq.history(conn, cfg, "SYN003", pd.Timestamp("2018-05-31"))
    after = eq.history(conn, cfg, "SYN003", pd.Timestamp("2018-06-30"))
    b = before[before["period_end"] == "2016-12-31"].iloc[0]
    a = after[after["period_end"] == "2016-12-31"].iloc[0]
    assert b["version_type"] == "original" and a["version_type"] == "amendment"
    assert a["operating_cash_flow"] == pytest.approx(b["operating_cash_flow"] * 3)
    assert (after["available_date"] <= "2018-06-30").all()
