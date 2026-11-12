import json
from pathlib import Path

import pandas as pd

EXPECTED = ["report.html", "run_summary.json", "screen_composite_latest.csv", "screen_rule_based_latest.csv",
            "backtest_summary.csv", "backtest_monthly.csv", "decile_returns.csv", "ic_summary.csv",
            "quality_checks.csv", "charts/cumulative_returns.png", "charts/pe_vs_forward_return.png",
            "charts/decile_excess_12m.png"]


def test_demo_run_writes_all_outputs(demo_run):
    out = Path(demo_run["project"]["output_dir"])
    missing = [f for f in EXPECTED if not (out / f).exists()]
    assert not missing


def test_screen_is_ranked_and_labelled(demo_run):
    out = Path(demo_run["project"]["output_dir"])
    screen = pd.read_csv(out / "screen_composite_latest.csv")
    assert screen["composite"].is_monotonic_decreasing
    assert set(screen["label"]) <= {"Research candidate", "Neutral", "Lower priority"}
    assert abs((screen["label"] == "Research candidate").mean() - 0.10) < 0.02
    rule = pd.read_csv(out / "screen_rule_based_latest.csv")
    assert rule["ev_ebit"].is_monotonic_increasing
    assert (rule["momentum_12_1"] > 0).all() and (rule["sector"] != "Financials").all()


def test_report_marks_synthetic_data(demo_run):
    out = Path(demo_run["project"]["output_dir"])
    assert "Synthetic demo data" in (out / "report.html").read_text(encoding="utf-8")
    assert json.loads((out / "run_summary.json").read_text())["mode"] == "demo"


def test_run_is_logged(demo_conn):
    runs = pd.read_sql_query("SELECT * FROM pipeline_runs", demo_conn)
    assert (runs["status"] == "success").any()


def test_short_history_skips_model_but_still_reports(fresh_demo_db, tmp_path):
    """Free sources often give ~4 years of statements; the pipeline must degrade gracefully."""
    from eqscreen import pipeline
    from eqscreen.config import ensure_dirs
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    conn.execute("DELETE FROM fundamentals WHERE period_end < '2021-01-01'")
    conn.commit()
    ensure_dirs(cfg)
    ctx = {"cfg": cfg, "conn": conn, "logger": get_logger(), "run_id": "short-history"}
    for stage in ("factors", "screen", "backtest", "model", "report"):
        pipeline.STAGES[stage](ctx)
    out = Path(cfg["project"]["output_dir"])
    status = json.loads((out / "model_status.json").read_text())
    assert status["status"] == "skipped"
    assert "Skipping model stage" in (out / "report.html").read_text(encoding="utf-8")


def test_analyst_note_uses_point_in_time_statement(demo_run, demo_conn):
    from eqscreen.note import build_note

    screen = pd.read_csv(Path(demo_run["project"]["output_dir"]) / "screen_composite_latest.csv")
    joiners = set(pd.read_sql_query("SELECT ticker FROM membership WHERE start_date IS NOT NULL", demo_conn)["ticker"])
    top = screen[~screen["ticker"].isin(joiners)].iloc[0]       # a member over the whole sample
    text = build_note(demo_run, top["ticker"])
    assert text.startswith("> **SYNTHETIC DEMO DATA.**")
    for heading in ("Where it sits in the screen", "Valuation against peers", "Financial profile",
                    "Screen history", "Price risk", "Filings behind these numbers", "Assumptions and limits"):
        assert f"## {heading}" in text
    assert f"ranked {int(top['rank'])} of {len(screen)}" in text
    assert f"public from {top['fundamentals_available']}" in text
    # An earlier as-of date uses the statement known then, not the latest one.
    early = build_note(demo_run, top["ticker"], as_of="2015-06")
    assert "Screen month-end: **2015-06-30**" in early and "(fiscal year to 2014-12-31)" in early


def test_old_database_gains_cash_flow_columns(tmp_path):
    import sqlite3
    from eqscreen.db import init_schema

    conn = sqlite3.connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE fundamentals (ticker TEXT, period_end TEXT, source TEXT, available_date TEXT, "
                 "available_date_is_estimated INTEGER, version_type TEXT, accession TEXT, revenue REAL, ebit REAL, "
                 "ebitda REAL, net_income REAL, eps_diluted REAL, shares_diluted REAL, total_equity REAL, "
                 "total_debt REAL, cash REAL, PRIMARY KEY (ticker, period_end, source, available_date))")
    conn.execute("INSERT INTO fundamentals (ticker, period_end, source, available_date, net_income) "
                 "VALUES ('A', '2020-12-31', 'sec', '2021-02-01', 5)")
    init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")}
    assert {"operating_cash_flow", "capex", "total_assets"} <= cols
    assert conn.execute("SELECT net_income FROM fundamentals").fetchone()[0] == 5


def test_run_summary_has_manifest(demo_run):
    m = json.loads((Path(demo_run["project"]["output_dir"]) / "run_summary.json").read_text())["manifest"]
    from eqscreen import __version__
    assert m["packages"]["eqscreen"] == __version__
    assert m["prices"]["rows"] > 0 and m["fundamentals"]["source"] == "synthetic"


def test_console_summary_lines(demo_run):
    from eqscreen.pipeline import format_summary

    lines = format_summary(demo_run, "run-x", 12.0)
    text = "\n".join(lines)
    assert lines[0] == "Run run-x finished (demo, 12s)"
    assert "research candidates" in text and "Retrospective backtest" in text and "Full log:" in text
    assert "Survivorship:" in text
    assert len(lines) <= 8


def test_shared_run_id_records_one_lineage_row(tmp_path):
    """Airflow runs one task per stage; with a shared run id they form one pipeline_runs row."""
    from eqscreen.pipeline import run
    from tests.conftest import _write_cfg

    cfg_path = _write_cfg(tmp_path)
    run(mode="demo", stages=["synthetic"], config_path=str(cfg_path), run_id="airflow-test-1")
    run(mode="demo", stages=["quality"], config_path=str(cfg_path), run_id="airflow-test-1")
    import sqlite3
    rows = sqlite3.connect(tmp_path / "demo.db").execute(
        "SELECT stages, status FROM pipeline_runs WHERE run_id = 'airflow-test-1'").fetchall()
    assert rows == [("synthetic,quality", "success")]


# --------------------------------------------------------------------------- retries (audit P2)
def _runs(tmp_path, run_id):
    import sqlite3
    c = sqlite3.connect(tmp_path / "demo.db")
    run_row = c.execute("SELECT status, message FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
    stages = c.execute("SELECT stage, attempt, status FROM pipeline_stage_runs WHERE run_id = ? ORDER BY id",
                       (run_id,)).fetchall()
    return run_row, stages


def test_successful_retry_clears_the_failure(tmp_path, monkeypatch):
    import pytest
    from eqscreen import pipeline
    from tests.conftest import _write_cfg

    cfg_path = str(_write_cfg(tmp_path))
    pipeline.run(mode="demo", stages=["synthetic"], config_path=cfg_path, run_id="r1")
    real = pipeline.STAGES["quality"]
    monkeypatch.setitem(pipeline.STAGES, "quality", lambda ctx: (_ for _ in ()).throw(RuntimeError("provider down")))
    with pytest.raises(RuntimeError):
        pipeline.run(mode="demo", stages=["quality"], config_path=cfg_path, run_id="r1")
    assert _runs(tmp_path, "r1")[0] == ("failed", "quality: RuntimeError: provider down")

    monkeypatch.setitem(pipeline.STAGES, "quality", real)                   # the retry succeeds
    pipeline.run(mode="demo", stages=["quality"], config_path=cfg_path, run_id="r1")
    run_row, stages = _runs(tmp_path, "r1")
    assert run_row == ("success", None)
    assert stages == [("synthetic", 1, "success"), ("quality", 1, "failed"), ("quality", 2, "success")]


def test_another_stage_succeeding_does_not_hide_a_failure(tmp_path, monkeypatch):
    import pytest
    from eqscreen import pipeline
    from tests.conftest import _write_cfg

    cfg_path = str(_write_cfg(tmp_path))
    pipeline.run(mode="demo", stages=["synthetic"], config_path=cfg_path, run_id="r2")
    monkeypatch.setitem(pipeline.STAGES, "quality", lambda ctx: (_ for _ in ()).throw(RuntimeError("bad data")))
    with pytest.raises(RuntimeError):
        pipeline.run(mode="demo", stages=["quality"], config_path=cfg_path, run_id="r2")
    pipeline.run(mode="demo", stages=["factors"], config_path=cfg_path, run_id="r2")
    run_row, _ = _runs(tmp_path, "r2")
    assert run_row[0] == "failed" and "quality" in run_row[1]


def test_single_run_failure_is_recorded(tmp_path, monkeypatch):
    import pytest
    from eqscreen import pipeline
    from tests.conftest import _write_cfg

    cfg_path = str(_write_cfg(tmp_path))
    monkeypatch.setitem(pipeline.STAGES, "quality", lambda ctx: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        pipeline.run(mode="demo", stages=["synthetic", "quality", "factors"], config_path=cfg_path, run_id="r3")
    run_row, stages = _runs(tmp_path, "r3")
    assert run_row == ("failed", "quality: ValueError: x")
    assert stages == [("synthetic", 1, "success"), ("quality", 1, "failed")]       # factors never started


def test_run_summary_estimated_share_ignores_tickers_outside_the_universe(fresh_demo_db):
    from eqscreen import pipeline
    from eqscreen.config import ensure_dirs
    from eqscreen.db import upsert_df
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    ensure_dirs(cfg)
    ctx = {"cfg": cfg, "conn": conn, "logger": get_logger(), "run_id": "est"}
    for stage in ("factors", "screen"):
        pipeline.STAGES[stage](ctx)
    path = Path(cfg["project"]["output_dir"]) / "run_summary.json"
    before = json.loads(path.read_text())["filing_dates_estimated_share"]
    upsert_df(conn, "fundamentals", pd.DataFrame([{"ticker": f"OLD{i}", "period_end": "2020-12-31", "source": "synthetic",
                                                   "available_date": "2021-03-31", "available_date_is_estimated": 1,
                                                   "version_type": "original", "net_income": 1.0} for i in range(500)]))
    pipeline.STAGES["screen"](ctx)
    assert json.loads(path.read_text())["filing_dates_estimated_share"] == before
