"""Pipeline orchestration and command-line entry point.

    python -m eqscreen.pipeline --mode demo                  # synthetic, offline, end to end
    python -m eqscreen.pipeline --mode live                  # fetch + compute + report
    python -m eqscreen.pipeline --mode live --from-stage quality   # recompute without refetching
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .config import ROOT, ensure_dirs, load_config
from .db import connect, init_schema, table_df
from .factors import RAW_FACTORS, build_factor_panel, load_panel, price_panel, save_panel
from .logging_utils import get_logger, new_run_id, utc_now

LIVE_STAGES = ["universe", "prices", "fundamentals", "quality", "factors", "screen", "backtest", "model", "report"]
DEMO_STAGES = ["synthetic", "quality", "factors", "screen", "backtest", "model", "report"]


# --------------------------------------------------------------------------- stages
def stage_synthetic(ctx):
    from .synthetic import load_into_db

    demo = ctx["cfg"].get("demo", {})
    counts = load_into_db(ctx["conn"], n_tickers=int(demo.get("n_tickers", 150)), seed=int(demo.get("seed", 7)))
    ctx["logger"].info("Synthetic data loaded: %s", counts)


def stage_universe(ctx):
    from .universe import build_universe

    build_universe(ctx["cfg"], ctx["conn"], ctx["run_id"], ctx["logger"])


def _tickers(conn) -> list[str]:
    return table_df(conn, "SELECT ticker FROM securities ORDER BY ticker")["ticker"].tolist()


def stage_prices(ctx):
    from .prices import fetch_prices

    fetch_prices(ctx["cfg"], _tickers(ctx["conn"]), ctx["conn"], ctx["run_id"], ctx["logger"])


def stage_fundamentals(ctx):
    from .fundamentals import fetch_fundamentals

    fetch_fundamentals(ctx["cfg"], _tickers(ctx["conn"]), ctx["conn"], ctx["run_id"], ctx["logger"])


def stage_quality(ctx):
    from .quality import persist_and_enforce, run_checks

    checks = run_checks(ctx["cfg"], ctx["conn"])
    df = persist_and_enforce(ctx["cfg"], ctx["conn"], ctx["run_id"], checks, ctx["logger"])
    df.to_csv(Path(ctx["cfg"]["project"]["output_dir"]) / "quality_checks.csv", index=False)


def stage_factors(ctx):
    from .factors import ev_coverage

    panel = build_factor_panel(ctx["conn"], ctx["cfg"])
    save_panel(ctx["conn"], panel)
    cov = ev_coverage(panel, ctx["cfg"])
    cov.to_csv(Path(ctx["cfg"]["project"]["output_dir"]) / "ev_coverage.csv", index=False, float_format="%.6g")
    if len(cov):
        last = cov.iloc[-1]
        ctx["logger"].info("EV factor inputs at %s: %d of %d non-excluded stocks complete (%d missing debt, "
                           "%d missing cash); policy=%s", last["month_end"].date(), last["ev_inputs_complete"],
                           last["with_fundamentals"], last["missing_debt"], last["missing_cash"],
                           ctx["cfg"]["factors"].get("ev_missing_policy", "exclude"))
    scored = panel["composite"].notna()
    ctx["logger"].info("Factor panel: %d rows, %d scored stock-months across %d months",
                       len(panel), int(scored.sum()), panel.loc[scored, "month_end"].nunique())


def stage_screen(ctx):
    from .screen import write_screens
    from .universe import survivorship_summary

    cfg, conn = ctx["cfg"], ctx["conn"]
    panel = load_panel(conn)
    info = write_screens(cfg, panel)
    # Current universe only, like the quality checks.
    est = table_df(conn, "SELECT AVG(f.available_date_is_estimated) AS s FROM fundamentals f "
                         "JOIN securities s ON s.ticker = f.ticker WHERE f.source = ?",
                   (cfg["fundamentals"]["source"],))["s"].iloc[0]
    summary = {
        "run_id": ctx["run_id"], "mode": cfg["mode"], "generated_at_utc": utc_now(), **info,
        "n_universe": len(_tickers(conn)), "fundamentals_source": cfg["fundamentals"]["source"],
        "filing_dates_estimated_share": None if pd.isna(est) else float(est),
        "manifest": run_manifest(cfg, conn),
        "survivorship": survivorship_summary(conn),
    }
    (Path(cfg["project"]["output_dir"]) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ctx["logger"].info("Screen as of %s: %d scored, %d research candidates, %d rule-based picks",
                       info["as_of"], info["n_scored"], info["n_candidates"], info["n_rule"])


def run_manifest(cfg: dict, conn) -> dict:
    """What a run depended on, so a result can be traced to its data and code versions."""
    import hashlib
    import importlib.metadata as md
    import platform

    def one(sql, params=()):
        return conn.execute(sql, params).fetchone()

    src = cfg["fundamentals"]["source"]
    px = one("SELECT COUNT(*), MIN(month_end), MAX(month_end) FROM prices_monthly")
    fu = one("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(period_end), MAX(available_date) FROM fundamentals "
             "WHERE source = ?", (src,))
    fl = one("SELECT MIN(fetched_at_utc), MAX(fetched_at_utc) FROM fetch_log")
    versions = {}
    for pkg in ("pandas", "numpy", "scikit-learn", "yfinance"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    from . import __version__
    versions["eqscreen"] = __version__
    cfg_hash = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != "project"}, sort_keys=True,
                                         default=str).encode()).hexdigest()[:12]
    return {
        "python": platform.python_version(), "packages": versions, "config_sha256_12": cfg_hash,
        "prices": {"rows": px[0], "first_month": px[1], "last_month": px[2]},
        "fundamentals": {"source": src, "rows": fu[0], "tickers": fu[1], "first_period": fu[2],
                         "last_available": fu[3]},
        "fetch_window_utc": {"first": fl[0], "last": fl[1]},
    }


def _backtest_inputs(ctx):
    conn = ctx["conn"]
    panel = load_panel(conn)
    fwd = bt.forward_return_matrix(price_panel(conn))
    ctx["eligible"] = (panel.pivot(index="month_end", columns="ticker", values="is_member").astype(float) > 0
                       if "is_member" in panel and (panel["is_member"] == 0).any() else None)
    return panel, fwd, _tickers(conn)


def _run_one(ctx, panel, fwd, universe, score, holding, min_names, lag=None, name=None, missing="cash"):
    b = ctx["cfg"]["backtest"]
    lag = int(b.get("execution_lag_months", 1)) if lag is None else int(lag)
    res = bt.run_backtest(panel, fwd, universe, ctx["cfg"]["prices"]["benchmark"], score,
                          top_q=float(b["top_quantile"]), holding=int(holding), cost_bps=float(b["cost_bps"]),
                          min_names=min_names, min_holdings=int(b["min_holdings"]), lag=lag,
                          missing_return=missing, eligible=ctx.get("eligible"))
    score = name or score
    if res.empty:
        ctx["logger"].warning("Backtest %s (hold %dm): no eligible months", score, holding)
        return None, None
    summ = bt.summarize(res, float(b["risk_free_annual"])).rename_axis("series").reset_index()
    summ.insert(0, "execution_lag", lag)
    summ.insert(0, "holding", int(holding))
    summ.insert(0, "strategy", score)
    res.insert(0, "execution_lag", lag)
    res.insert(0, "holding", int(holding))
    res.insert(0, "strategy", score)
    return summ, res


def alternative_lag(cfg: dict) -> int:
    """The execution convention not used for the headline: same-close if lagged, else one month."""
    return 0 if int(cfg["backtest"].get("execution_lag_months", 1)) > 0 else 1


def stage_backtest(ctx):
    cfg, logger = ctx["cfg"], ctx["logger"]
    out = Path(cfg["project"]["output_dir"])
    panel, fwd, universe = _backtest_inputs(ctx)
    min_names = int(cfg["factors"]["min_names_per_date"])
    min_rule = max(int(cfg["backtest"]["min_holdings"]), min_names // 3)

    plan = [("composite", h, min_names) for h in cfg["backtest"]["holding_months"]]
    plan += [("rule_score", 1, min_rule), ("value_score", 1, min_names), ("momentum_score", 1, min_names)]
    if cfg["factors"]["weights"].get("quality", 0) > 0:
        plan.append(("quality_score", 1, min_names))
    summaries, monthlies = [], []
    for score, h, mn in plan:
        s, m = _run_one(ctx, panel, fwd, universe, score, h, mn)
        if s is not None:
            summaries.append(s)
            monthlies.append(m)
    # Robustness: the other execution convention, and a worst case for missing returns.
    head_h = cfg["backtest"]["holding_months"][0]
    alt_lag = alternative_lag(cfg)
    for kwargs in ({"lag": alt_lag, "name": f"composite_lag{alt_lag}"},
                   {"missing": "loss", "name": "composite_missing_loss"}):
        s, m = _run_one(ctx, panel, fwd, universe, "composite", head_h, min_names, **kwargs)
        if s is not None:
            summaries.append(s)
            monthlies.append(m)
    if not summaries:
        raise RuntimeError("No backtest could run: not enough scored months. Check coverage and min_names_per_date.")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "backtest_summary.csv", index=False, float_format="%.6g")
    pd.concat(monthlies, ignore_index=True).to_csv(out / "backtest_monthly.csv", index=False, float_format="%.6g")

    horizons = [int(h) for h in cfg["backtest"]["forward_horizons"]]
    deciles = []
    for score in ("composite", "value_score", "momentum_score", "rule_score"):
        d = bt.decile_returns(panel, score, horizons, min_names=min_rule if score == "rule_score" else min_names)
        if len(d):
            deciles.append(d.assign(score=score))
    pd.concat(deciles, ignore_index=True).to_csv(out / "decile_returns.csv", index=False, float_format="%.6g")

    ic_rows, ic_ts = [], []
    scores = (["composite", "value_score", "momentum_score", "quality_score", "earnings_quality_score", "rule_score"]
              + [f"z_{c}" for c in RAW_FACTORS] + ["fcf_yield"])
    for score in scores:
        for h in horizons:
            s = bt.ic_series(panel, score, f"ret_fwd_{h}m")
            ic_rows.append({"score": score, "horizon_months": h, **bt.ic_summary(s, h)})
            ic_ts.append(pd.DataFrame({"month_end": s.index, "ic": s.values, "score": score, "horizon_months": h}))
    pd.DataFrame(ic_rows).to_csv(out / "ic_summary.csv", index=False, float_format="%.6g")
    pd.concat(ic_ts, ignore_index=True).to_csv(out / "ic_timeseries.csv", index=False, float_format="%.6g")

    head = summary[(summary["strategy"] == "composite") & (summary["series"] == "strategy_net")].iloc[0]
    logger.info("Backtest composite (hold %dm, lag %dm, avg %.0f holdings) %s to %s: CAGR %.2f%%, Sharpe %.2f, "
                "excess vs EW universe %.2f%%/yr (Newey-West t %.2f)", head["holding"], head["execution_lag"],
                head["avg_holdings"], head["start"], head["end"], 100 * head["cagr"], head["sharpe"],
                100 * head["excess_return_ann"], head["t_stat_excess"])


def stage_model(ctx):
    from .model import cross_validate

    cfg, logger = ctx["cfg"], ctx["logger"]
    out = Path(cfg["project"]["output_dir"])
    panel, fwd, universe = _backtest_inputs(ctx)
    res = cross_validate(panel, cfg, logger)
    status = {"status": res["status"], "message": res.get("message", "")}
    if res["status"] == "ok":
        res["cv"].to_csv(out / "model_cv.csv", index=False, float_format="%.6g")
        res["coefficients"].rename_axis("feature").reset_index().to_csv(out / "model_coefficients.csv", index=False)
        res["pca_loadings"].rename_axis("feature").reset_index().to_csv(out / "pca_loadings.csv", index=False)
        res["pca_explained"].rename_axis("component").reset_index().to_csv(out / "pca_explained.csv", index=False)

        # Out-of-sample backtest of the model score vs the composite on the same stock-months.
        from .model import same_cohort_panel
        p = same_cohort_panel(panel, res["oos_scores"])
        min_names = int(cfg["factors"]["min_names_per_date"])
        extra_s, extra_m = [], []
        for score in ("ml_logit_oos", "composite_same_cohort"):
            s, m = _run_one(ctx, p, fwd, universe, score, 1, min_names)
            if s is not None:
                extra_s.append(s)
                extra_m.append(m)
        if extra_s:
            summ_path, mon_path = out / "backtest_summary.csv", out / "backtest_monthly.csv"
            keep = lambda df: df[~df["strategy"].isin(["ml_logit_oos", "composite_same_cohort"])]  # noqa: E731
            pd.concat([keep(pd.read_csv(summ_path)), *extra_s]).to_csv(summ_path, index=False, float_format="%.6g")
            pd.concat([keep(pd.read_csv(mon_path)), *extra_m]).to_csv(mon_path, index=False, float_format="%.6g")
        status["folds"] = len(res["cv"])
    (out / "model_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def stage_report(ctx):
    from .report import build_report

    out = Path(ctx["cfg"]["project"]["output_dir"])
    summ_path = out / "run_summary.json"
    run = json.loads(summ_path.read_text(encoding="utf-8"))
    run["run_id"] = ctx["run_id"]
    summ_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    path = build_report(ctx["cfg"], ctx["conn"])
    ctx["logger"].info("Report written to %s", path)


STAGES = {name: globals()[f"stage_{name}"] for name in set(LIVE_STAGES + DEMO_STAGES)}


# --------------------------------------------------------------------------- runner
def format_summary(cfg: dict, run_id: str, seconds: float) -> list[str]:
    """A few lines for the console at the end of a run."""
    out = Path(cfg["project"]["output_dir"])
    lines = [f"Run {run_id} finished ({cfg['mode']}, {seconds:.0f}s)"]
    rs = out / "run_summary.json"
    if rs.exists():
        r = json.loads(rs.read_text(encoding="utf-8"))
        if r.get("as_of"):
            lines.append(f"  Screen as of {r['as_of']}: {r['n_scored']} stocks scored, "
                         f"{r.get('n_candidates', 0)} research candidates")
        sv = r.get("survivorship", {})
        if sv.get("membership_data"):
            lines.append(f"  Survivorship: {sv['removed_missing_from_data']} of {sv['removed_since_first_price']} "
                         "index departures have no data (results biased upward)")
    bs = out / "backtest_summary.csv"
    if bs.exists():
        s = pd.read_csv(bs)
        h = int(cfg["backtest"]["holding_months"][0])
        net = s[(s["strategy"] == "composite") & (s["holding"] == h) & (s["series"] == "strategy_net")]
        ew = s[(s["strategy"] == "composite") & (s["holding"] == h) & (s["series"] == "ew_universe")]
        if len(net) and len(ew):
            n, e = net.iloc[0], ew.iloc[0]
            lines.append(f"  Retrospective backtest {n['start']} to {n['end']}: {n['cagr']:.1%} a year vs "
                         f"{e['cagr']:.1%} equal-weight, Sharpe {n['sharpe']:.2f}, Newey-West t {n['t_stat_excess']:.1f}")
    qc = out / "quality_checks.csv"
    if qc.exists():
        q = pd.read_csv(qc)
        warns = q[q["status"] == "warn"]["name"].tolist()
        if warns:
            lines.append(f"  Quality warnings: {', '.join(warns)}")
    if (out / "report.html").exists():
        lines.append(f"  Report: {out / 'report.html'}")
    lines.append(f"  Full log: {Path(cfg['project']['log_dir']) / 'pipeline.log'}")
    return lines


def run_status(conn, run_id: str) -> tuple[str, str | None]:
    """Status of a run from the latest attempt of each of its stages.

    failed if any stage's latest attempt failed (message: that error); running if any is still
    running; success otherwise. A retry that succeeds therefore clears an earlier failure.
    """
    latest = conn.execute(
        "SELECT stage, status, message FROM pipeline_stage_runs WHERE id IN "
        "(SELECT MAX(id) FROM pipeline_stage_runs WHERE run_id = ? GROUP BY stage) ORDER BY id", (run_id,)).fetchall()
    failed = [(s, m) for s, st, m in latest if st == "failed"]
    if failed:
        return "failed", "; ".join(f"{s}: {m}" for s, m in failed)[:1000]
    if any(st == "running" for _, st, _ in latest):
        return "running", None
    return "success", None


def run(mode: str = "demo", stages: list[str] | None = None, config_path: str | None = None,
        from_stage: str | None = None, verbose: bool | None = None, run_id: str | None = None) -> str:
    """Run pipeline stages. Pass `run_id` to record several invocations (e.g. one Airflow task
    per stage) under one run: the stages are appended to the same pipeline_runs row."""
    cfg = load_config(config_path, mode=mode)
    ensure_dirs(cfg)
    if verbose is None:
        verbose = cfg.get("logging", {}).get("console", "summary") == "verbose"
    logger = get_logger(cfg["project"]["log_dir"], console_level="INFO" if verbose else "WARNING")
    order = DEMO_STAGES if mode == "demo" else LIVE_STAGES
    if stages:
        bad = [s for s in stages if s not in order]
        if bad:
            raise ValueError(f"Unknown stages for {mode} mode: {bad}. Valid: {order}")
        todo = [s for s in order if s in stages]
    elif from_stage:
        todo = order[order.index(from_stage):]
    else:
        todo = order

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    run_id = run_id or new_run_id()
    conn = connect(cfg["project"]["db_path"])
    init_schema(conn)
    existing = conn.execute("SELECT stages FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
    if existing:
        seen = [x for x in (existing[0] or "").split(",") if x]
        stages_so_far = ",".join(seen + [s for s in todo if s not in seen])
        conn.execute("UPDATE pipeline_runs SET stages = ?, status = 'running' WHERE run_id = ?",
                     (stages_so_far, run_id))
    else:
        conn.execute("INSERT INTO pipeline_runs (run_id, started_at_utc, mode, stages, status) VALUES (?, ?, ?, ?, ?)",
                     (run_id, utc_now(), mode, ",".join(todo), "running"))
    conn.commit()
    ctx = {"cfg": cfg, "conn": conn, "logger": logger, "run_id": run_id}
    logger.info("Run %s started (%s mode): %s", run_id, mode, " > ".join(todo))
    status, message = "success", ""
    started = time.perf_counter()
    try:
        for i, name in enumerate(todo, start=1):
            t0 = time.perf_counter()
            logger.info("Stage %s ...", name)
            attempt = 1 + conn.execute("SELECT COUNT(*) FROM pipeline_stage_runs WHERE run_id = ? AND stage = ?",
                                       (run_id, name)).fetchone()[0]
            cur_stage = conn.execute("INSERT INTO pipeline_stage_runs (run_id, stage, attempt, started_at_utc, status) "
                                     "VALUES (?, ?, ?, ?, 'running')", (run_id, name, attempt, utc_now())).lastrowid
            conn.commit()
            if not verbose and name in ("universe", "prices", "fundamentals"):
                print(f"[{i}/{len(todo)}] {name}: downloading (this can take a while) ...", flush=True)
            try:
                STAGES[name](ctx)
            except Exception as exc:
                conn.execute("UPDATE pipeline_stage_runs SET finished_at_utc = ?, status = 'failed', message = ? "
                             "WHERE id = ?", (utc_now(), f"{type(exc).__name__}: {exc}"[:1000], cur_stage))
                conn.commit()
                raise
            conn.execute("UPDATE pipeline_stage_runs SET finished_at_utc = ?, status = 'success' WHERE id = ?",
                         (utc_now(), cur_stage))
            conn.commit()
            logger.info("Stage %s done in %.1fs", name, time.perf_counter() - t0)
            if not verbose:
                print(f"[{i}/{len(todo)}] {name:<13} done ({time.perf_counter() - t0:.1f}s)", flush=True)
    except Exception as exc:
        status, message = "failed", f"{type(exc).__name__}: {exc}"
        logger.error("Run %s failed: %s\n%s", run_id, message, traceback.format_exc())
        raise
    finally:
        overall, overall_msg = run_status(conn, run_id)
        conn.execute("UPDATE pipeline_runs SET finished_at_utc = ?, status = ?, message = ? WHERE run_id = ?",
                     (utc_now(), overall, overall_msg, run_id))
        conn.commit()
        conn.close()
    logger.info("Run %s finished: %s", run_id, status)
    if not verbose:
        print("\n".join(format_summary(cfg, run_id, time.perf_counter() - started)))
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Equity screening & valuation factor pipeline")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="show the full log on the console")
    parser.add_argument("--run-id", default=None, help="record this invocation under an existing/shared run id")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stages", default=None, help="comma-separated subset of stages to run")
    group.add_argument("--from-stage", default=None, help="run this stage and everything after it")
    args = parser.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    run(args.mode, stages, args.config, args.from_stage, verbose=True if args.verbose else None, run_id=args.run_id)


if __name__ == "__main__":
    main()
