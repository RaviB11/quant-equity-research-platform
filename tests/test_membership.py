import numpy as np
import pandas as pd
import pytest

from eqscreen import backtest as bt
from eqscreen.factors import load_panel
from eqscreen.universe import parse_wikipedia_changes, reconstruct_membership, survivorship_summary

CHANGES_HTML = """<table id="changes">
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>June 1, 2020</td><td>CCC</td><td>C Corp</td><td>XXX</td><td>X Corp</td><td>Market cap change</td></tr>
<tr><td>March 1, 2018</td><td>XXX</td><td>X Corp</td><td>YYY</td><td>Y Corp</td><td>Acquired</td></tr>
<tr><td>January 5, 2017</td><td></td><td></td><td>BRK.A</td><td>Old Class</td><td>Delisted</td></tr>
</table>"""


def test_parse_changes_table_with_two_row_header():
    ch = parse_wikipedia_changes(CHANGES_HTML)
    assert ch["change_date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-06-01", "2018-03-01", "2017-01-05"]
    assert ch["added"].tolist()[:2] == ["CCC", "XXX"] and ch["added"].isna().iloc[2]
    assert ch["removed"].tolist() == ["XXX", "YYY", "BRK-A"]


def test_reconstruct_membership_walks_back_from_today():
    spans = reconstruct_membership(["AAA", "CCC"], parse_wikipedia_changes(CHANGES_HTML)).set_index("ticker")
    assert spans.loc["AAA"].isna().all()                                   # member throughout
    assert spans.loc["CCC", "start_date"] == "2020-06-01" and pd.isna(spans.loc["CCC", "end_date"])
    assert (spans.loc["XXX", "start_date"], spans.loc["XXX", "end_date"]) == ("2018-03-01", "2020-05-31")
    assert pd.isna(spans.loc["YYY", "start_date"]) and spans.loc["YYY", "end_date"] == "2018-02-28"


def test_non_members_are_not_scored_and_leave_the_benchmark(demo_run, demo_conn):
    panel = load_panel(demo_conn)
    joined = pd.read_sql_query("SELECT ticker, start_date FROM membership WHERE start_date IS NOT NULL", demo_conn)
    x = panel.merge(joined, on="ticker")
    before = x[x["month_end"] < pd.to_datetime(x["start_date"])]
    assert len(before) > 100
    assert (before["is_member"] == 0).all() and before["composite"].isna().all() and before["rule_score"].isna().all()
    assert x.loc[x["month_end"] >= pd.to_datetime(x["start_date"]), "composite"].notna().mean() > 0.9


def test_equal_weight_benchmark_respects_eligibility():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    panel = pd.DataFrame([{"month_end": d, "ticker": t, "s": v} for d in dates for t, v in (("A", 2), ("B", 1))])
    fwd = pd.DataFrame([[0.1, 0.5], [0.1, 0.5], [np.nan, np.nan]], index=dates, columns=["A", "B"])
    eligible = pd.DataFrame([[True, False], [True, True], [True, True]], index=dates, columns=["A", "B"])
    res = bt.run_backtest(panel, fwd, ["A", "B"], None, "s", top_q=0.5, min_names=2, min_holdings=1,
                          cost_bps=0, eligible=eligible)
    assert res["ew_universe"].tolist() == pytest.approx([0.1, 0.3])


def test_survivorship_summary_counts_missing_departures(demo_run, demo_conn):
    sv = survivorship_summary(demo_conn)
    assert sv["membership_data"] and sv["removed_since_first_price"] == 12
    assert sv["removed_missing_from_data"] == 12 and sv["removed_with_price_data"] == 0


# --------------------------------------------------------------------------- source switches (audit P2-C)
def _fresh_db(tmp_path):
    from eqscreen.db import connect, init_schema
    conn = connect(str(tmp_path / "u.db"))
    init_schema(conn)
    return conn


def _live_cfg(**universe):
    from eqscreen.config import load_config
    cfg = load_config(mode="live")
    cfg["universe"].update(universe)
    cfg["project"]["raw_dir"] = cfg["project"]["raw_dir"]
    return cfg


def test_switching_to_csv_universe_clears_stale_membership(tmp_path):
    from eqscreen.logging_utils import get_logger
    from eqscreen.universe import build_universe

    conn = _fresh_db(tmp_path)
    conn.execute("INSERT INTO membership VALUES ('AAPL', '2020-01-01', NULL)")
    conn.execute("INSERT INTO index_changes VALUES ('2020-01-01', 'AAPL', 'Apple', NULL, NULL)")
    conn.commit()
    build_universe(_live_cfg(source="file"), conn, "r", get_logger())
    assert conn.execute("SELECT COUNT(*) FROM membership").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM index_changes").fetchone()[0] == 0


def test_failed_change_table_parse_does_not_leave_old_membership(tmp_path, monkeypatch):
    from eqscreen import universe as u
    from eqscreen.logging_utils import get_logger

    conn = _fresh_db(tmp_path)
    conn.execute("INSERT INTO membership VALUES ('OLD', NULL, '2019-01-01')")
    conn.commit()
    html = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
    <th>GICS Sub-Industry</th><th>CIK</th></tr><tr><td>AAA</td><td>A Co</td><td>Industrials</td><td>M</td><td>1</td></tr>
    </table>"""                                                # no changes table on the page

    class R:
        text = html
        def raise_for_status(self):
            pass
    monkeypatch.setattr(u.requests, "get", lambda *a, **k: R())
    cfg = _live_cfg(source="sp500_wikipedia")
    cfg["project"]["raw_dir"] = str(tmp_path)
    u.build_universe(cfg, conn, "r", get_logger())
    assert conn.execute("SELECT COUNT(*) FROM membership").fetchone()[0] == 0
    assert [r[0] for r in conn.execute("SELECT ticker FROM securities")] == ["AAA"]
