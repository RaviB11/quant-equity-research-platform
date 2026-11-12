"""Static HTML report (Plotly) and PNG example charts (matplotlib) from pipeline outputs."""
from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from .factors import load_panel  # noqa: E402

C = {"ink": "#14213D", "muted": "#5B6575", "rule": "#D5DAE1", "strategy": "#0E6E6E",
     "universe": "#8C95A3", "bench": "#B07A12", "pos": "#1F7A4D", "neg": "#B03A2E"}
FONT = "Public Sans, system-ui, -apple-system, Segoe UI, sans-serif"

STRATEGY_NAMES = {"composite": "Composite (value + momentum)", "rule_score": "Rule: cheap EV/EBIT, positive momentum",
                  "value_score": "Value only", "momentum_score": "Momentum only", "quality_score": "Quality only",
                  "ml_logit_oos": "Logistic model (out-of-sample)", "composite_same_cohort": "Composite, same stock-months as model",
                  "composite_lag1": "Composite, traded one month after signal",
                  "composite_lag0": "Composite, same-close trading (optimistic)",
                  "composite_missing_loss": "Composite, missing returns as total loss",
                  "z_earnings_yield": "Earnings yield (E/P)", "z_ebit_ev": "EBIT / EV", "z_ebitda_ev": "EBITDA / EV",
                  "z_book_to_market": "Book / market", "z_momentum_12_1": "Momentum 12-1", "z_roe": "Return on equity",
                  "earnings_quality_score": "Earnings quality only", "z_accruals_low": "Low accruals",
                  "fcf_yield": "FCF yield (not in composite)"}


# --------------------------------------------------------------------------- formatting
def pct(x, digits=1, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def num(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{digits}f}"


def _layout(fig: go.Figure, height=360, hovermode="x unified", **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=56, r=16, t=24, b=44), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12, color=C["muted"]),
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)), hovermode=hovermode, **kw)
    fig.update_xaxes(gridcolor="#EEF1F4", linecolor=C["rule"], zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F4", linecolor=C["rule"], zerolinecolor=C["rule"])
    return fig


def _fig_html(fig: go.Figure, first: bool, inline: bool = False) -> str:
    js = (True if inline else "cdn") if first else False
    return fig.to_html(full_html=False, include_plotlyjs=js, default_width="100%",
                       config={"displayModeBar": False, "responsive": True})


TEXT_FORMATTED = {"Label", "Status"}


def _table(df: pd.DataFrame, formats: dict, cls: str = "") -> str:
    numeric = {c for c in df.columns if c in formats and c not in TEXT_FORMATTED}
    head = "".join(f'<th{" class=n" if c in numeric else ""}>{html.escape(str(c))}</th>' for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            f = formats.get(c)
            text = f(v) if f else ("–" if pd.isna(v) else html.escape(str(v)))
            num_cls = ' class="n"' if c in numeric else ""
            cells.append(f"<td{num_cls}>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# --------------------------------------------------------------------------- charts
def chart_cumulative(monthly: pd.DataFrame) -> go.Figure:
    m = monthly.set_index("period_end").sort_index()
    fig = go.Figure()
    for col, name, color in (("net", "Composite top decile (after costs)", C["strategy"]),
                             ("ew_universe", "Equal-weight universe", C["universe"]),
                             ("benchmark", "Benchmark", C["bench"])):
        if col in m and m[col].notna().any():
            fig.add_trace(go.Scatter(x=m.index, y=(1 + m[col].fillna(0)).cumprod(), name=name,
                                     line=dict(color=color, width=2.4 if col == "net" else 1.6)))
    fig.update_yaxes(type="log", title="Growth of $1 (log scale)")
    return _layout(fig, 380)


def chart_deciles(dec: pd.DataFrame, horizon: int) -> go.Figure:
    d = dec[(dec["score"] == "composite") & (dec["horizon_months"] == horizon)].sort_values("decile")
    colors = [C["pos"] if v >= 0 else C["neg"] for v in d["avg_excess_return"]]
    fig = go.Figure(go.Bar(x=d["decile"].astype(str), y=d["avg_excess_return"], marker_color=colors,
                           hovertemplate="Decile %{x}: %{y:.2%}<extra></extra>"))
    fig.update_xaxes(title="Composite score decile (10 = most attractive)")
    fig.update_yaxes(title=f"Avg {horizon}-month return vs cross-section", tickformat=".1%")
    return _layout(fig, 320, hovermode="closest")


def pe_scatter_data(panel: pd.DataFrame, horizon: int = 12) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.DataFrame]:
    fwd = f"ret_fwd_{horizon}m"
    ok = panel[panel["pe"].between(0, 100) & panel[fwd].notna()]
    if ok.empty:
        return ok, None, pd.DataFrame()
    counts = ok.groupby("month_end").size()
    date = counts[counts >= 20].index.max() if (counts >= 20).any() else counts.index.max()
    cross = ok[ok["month_end"] == date]
    pooled = ok.assign(pe_bin=pd.qcut(ok["pe"], 10, duplicates="drop"))
    bins = pooled.groupby("pe_bin", observed=True).agg(median_pe=("pe", "median"), median_fwd=(fwd, "median"),
                                                       n=(fwd, "size")).reset_index(drop=True)
    return cross, date, bins


def chart_pe_scatter(cross, date, bins, horizon=12) -> go.Figure:
    fwd = f"ret_fwd_{horizon}m"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cross["pe"], y=cross[fwd], mode="markers", name=f"Stocks at {date:%b %Y}",
                             text=cross["ticker"], marker=dict(color=C["strategy"], size=7, opacity=0.55),
                             hovertemplate="%{text}<br>P/E %{x:.1f}<br>Next 12m %{y:.1%}<extra></extra>"))
    if len(bins):
        fig.add_trace(go.Scatter(x=bins["median_pe"], y=bins["median_fwd"], mode="lines+markers",
                                 name="All months, median by P/E decile", line=dict(color=C["bench"], width=2)))
    fig.update_xaxes(title="Trailing P/E", type="log")
    fig.update_yaxes(title="Forward 12-month total return", tickformat=".0%")
    return _layout(fig, 380, hovermode="closest")


def chart_ic(ic_ts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in (("composite", C["strategy"]), ("value_score", C["bench"]), ("momentum_score", C["universe"])):
        s = ic_ts[(ic_ts["score"] == col) & (ic_ts["horizon_months"] == 1)].set_index("month_end")["ic"].sort_index()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(12, min_periods=6).mean(),
                                     name=STRATEGY_NAMES.get(col, col), line=dict(color=color, width=2)))
    fig.add_hline(y=0, line_color=C["rule"])
    fig.update_yaxes(title="Rank IC vs next-month return (12m rolling mean)")
    return _layout(fig, 320)


def chart_corr(panel: pd.DataFrame) -> go.Figure:
    cols = ["z_earnings_yield", "z_ebit_ev", "z_ebitda_ev", "z_book_to_market", "z_momentum_12_1", "z_roe"]
    labels = ["Earnings yield", "EBIT/EV", "EBITDA/EV", "Book/market", "Momentum 12-1", "ROE"]
    mats = [g[cols].corr(method="spearman") for _, g in panel.groupby("month_end") if g[cols].notna().sum().min() >= 20]
    m = sum(mats) / len(mats) if mats else pd.DataFrame(np.nan, index=cols, columns=cols)
    fig = go.Figure(go.Heatmap(z=m.values, x=labels, y=labels, zmin=-1, zmax=1,
                               colorscale=[[0, C["neg"]], [0.5, "#FFFFFF"], [1, C["strategy"]]],
                               text=np.round(m.values, 2), texttemplate="%{text}", hoverinfo="skip"))
    fig = _layout(fig, 400, hovermode=False)
    fig.update_layout(margin=dict(l=110, r=16, t=24, b=80))
    return fig


def write_pngs(out: Path, monthly, dec, cross, date, bins, synthetic: bool) -> None:
    charts = out / "charts"
    charts.mkdir(exist_ok=True, parents=True)
    tag = "  [SYNTHETIC DEMO DATA]" if synthetic else ""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    m = monthly.set_index("period_end").sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col, name, color in (("net", "Composite top decile (net)", C["strategy"]),
                             ("ew_universe", "Equal-weight universe", C["universe"]), ("benchmark", "Benchmark", C["bench"])):
        if m[col].notna().any():
            ax.plot(m.index, (1 + m[col].fillna(0)).cumprod(), label=name, color=color, lw=2 if col == "net" else 1.3)
    ax.set_yscale("log")
    ax.set_title("Growth of $1" + tag, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(charts / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    d = dec[(dec["score"] == "composite") & (dec["horizon_months"] == 12)].sort_values("decile")
    if len(d):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(d["decile"].astype(str), d["avg_excess_return"],
               color=[C["pos"] if v >= 0 else C["neg"] for v in d["avg_excess_return"]])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Composite decile (10 = most attractive)")
        ax.set_title("Avg 12-month return vs cross-section, by decile" + tag, loc="left")
        fig.tight_layout()
        fig.savefig(charts / "decile_excess_12m.png", dpi=150)
        plt.close(fig)

    if date is not None and len(cross):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(cross["pe"], cross["ret_fwd_12m"], s=14, alpha=0.55, color=C["strategy"], label=f"Stocks at {date:%b %Y}")
        if len(bins):
            ax.plot(bins["median_pe"], bins["median_fwd"], color=C["bench"], marker="o", label="All months, median by P/E decile")
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Trailing P/E")
        ax.set_ylabel("Forward 12-month return")
        ax.set_title("P/E vs forward return" + tag, loc="left")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(charts / "pe_vs_forward_return.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- page
CSS = """
:root{--ink:#14213D;--muted:#5B6575;--rule:#D5DAE1;--wash:#F4F6F8;--paper:#FFFFFF;--teal:#0E6E6E;
--pos:#1F7A4D;--neg:#B03A2E;--ochre:#B07A12;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
.synthetic{background:repeating-linear-gradient(-45deg,#FFF4D6 0 14px,#FDEBB8 14px 28px);color:#5A4200;padding:.7rem 1.25rem;font-weight:600;border-bottom:1px solid #E9D28A}
main{max-width:1080px;margin:0 auto;padding:2.5rem 1.25rem 4rem}
header.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:.5rem 2rem;color:var(--muted);font-size:.9rem;border-bottom:2px solid var(--ink);padding-bottom:.6rem}
header.top strong{color:var(--ink)}
.meta{display:flex;flex-wrap:wrap;gap:.25rem 1.4rem}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;font-weight:600;letter-spacing:-.01em}
.verdict{font-family:"Source Serif 4",Georgia,serif;font-size:clamp(1.55rem,3.2vw,2.2rem);line-height:1.25;font-weight:500;max-width:30em;margin:1.8rem 0 1.4rem}
dl.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0;margin:0 0 2rem;border-top:1px solid var(--rule)}
dl.stats div{padding:.75rem .9rem .75rem 0;border-bottom:1px solid var(--rule)}
dl.stats dt{color:var(--muted);font-size:.85rem}
dl.stats dd{margin:.15rem 0 0;font-size:1.35rem;font-weight:600}
dl.stats dd small{display:block;font-size:.8rem;font-weight:400;color:var(--muted)}
section{margin-top:3rem}
h2{font-size:1.55rem;margin:0 0 .35rem}
h3{font-size:1.1rem;margin:1.6rem 0 .3rem}
p.lede{color:var(--muted);max-width:46em;margin:.2rem 0 1rem}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.5rem 2rem}
.tw{overflow-x:auto;margin:.4rem 0 1rem}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--ink);padding:.45rem .6rem .35rem 0;white-space:nowrap}
td{padding:.38rem .6rem .38rem 0;border-bottom:1px solid var(--rule);white-space:nowrap}
th.n,td.n{text-align:right}
table.narrow{max-width:460px}
tbody tr:nth-child(even){background:var(--wash)}
.pill{display:inline-block;padding:0 .45rem;border-radius:3px;font-size:.78rem;font-weight:600}
.buy{background:#DDF0E6;color:var(--pos)}.avoid{background:#F6E0DD;color:var(--neg)}.hold{background:var(--wash);color:var(--muted)}
.pass{color:var(--pos);font-weight:600}.warn{color:var(--ochre);font-weight:600}.fail{color:var(--neg);font-weight:600}
ul.notes{max-width:52em;padding-left:1.1rem}ul.notes li{margin:.35rem 0}
footer{margin-top:3.5rem;padding-top:1rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.85rem}
@media (max-width:860px){.grid2{grid-template-columns:1fr}}@media (max-width:600px){main{padding-top:1.5rem}}
"""


def _label_pill(v: str) -> str:
    cls = {"Research candidate": "buy", "Lower priority": "avoid"}.get(v, "hold")
    return f'<span class="pill {cls}">{html.escape(str(v))}</span>'


def build_report(cfg: dict, conn: sqlite3.Connection) -> Path:
    out = Path(cfg["project"]["output_dir"])
    synthetic = cfg["mode"] == "demo"
    run = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out / "backtest_summary.csv")
    monthly_all = pd.read_csv(out / "backtest_monthly.csv", parse_dates=["period_end", "formation_date"])
    dec = pd.read_csv(out / "decile_returns.csv")
    ic_sum = pd.read_csv(out / "ic_summary.csv")
    ic_ts = pd.read_csv(out / "ic_timeseries.csv", parse_dates=["month_end"])
    screen = pd.read_csv(out / "screen_composite_latest.csv")
    rule = pd.read_csv(out / "screen_rule_based_latest.csv")
    qc = pd.read_csv(out / "quality_checks.csv") if (out / "quality_checks.csv").exists() else pd.DataFrame()
    model_status = json.loads((out / "model_status.json").read_text(encoding="utf-8")) if (out / "model_status.json").exists() else {"status": "not run"}
    panel = load_panel(conn)

    head_h = int(cfg["backtest"]["holding_months"][0])
    monthly = monthly_all[(monthly_all["strategy"] == "composite") & (monthly_all["holding"] == head_h)]
    from .pipeline import alternative_lag
    head_lag = int(cfg["backtest"].get("execution_lag_months", 1))
    alt_lag = alternative_lag(cfg)
    lag_row = summary[(summary["strategy"] == f"composite_lag{alt_lag}") & (summary["series"] == "strategy_net")]
    miss_row = summary[(summary["strategy"] == "composite_missing_loss") & (summary["series"] == "strategy_net")]
    ev_cov = pd.read_csv(out / "ev_coverage.csv") if (out / "ev_coverage.csv").exists() else pd.DataFrame()
    srow = lambda strat, series: summary[(summary["strategy"] == strat) & (summary["holding"] == head_h)  # noqa: E731
                                         & (summary["series"] == series)].iloc[0]
    net, ew = srow("composite", "strategy_net"), srow("composite", "ew_universe")
    years = net["months"] / 12
    verb = "beat" if net["cagr"] > ew["cagr"] else "trailed"
    verdict = (f"In this retrospective test over {years:.1f} years to {pd.Timestamp(net['end']):%B %Y}, "
               f"the highest-scoring stocks "
               f"(about {net['avg_holdings']:.0f} names each month) {verb} the equal-weight universe: "
               f"{pct(net['cagr'])} a year after costs against {pct(ew['cagr'])}, "
               f"with a Sharpe ratio of {num(net['sharpe'])} against {num(ew['sharpe'])}.")

    stats = [("CAGR, net of costs", pct(net["cagr"]), f"universe {pct(ew['cagr'])}"),
             ("Excess return a year", pct(net["excess_return_ann"], signed=True),
              f"Newey-West t = {num(net['t_stat_excess'])}"),
             ("Sharpe ratio", num(net["sharpe"]), f"universe {num(ew['sharpe'])}"),
             ("Max drawdown", pct(net["max_drawdown"]), f"universe {pct(ew['max_drawdown'])}"),
             ("Information ratio", num(net["information_ratio"]), f"hit rate {pct(net['hit_rate'], 0)}"),
             (("Same-close trading" if alt_lag == 0 else "Traded a month later"),
              pct(lag_row.iloc[0]["cagr"]) if len(lag_row) else "–",
              "CAGR, optimistic convention" if alt_lag == 0 else "CAGR with a one-month delay")]
    stats_html = "".join(f"<div><dt>{a}</dt><dd>{b}<small>{c}</small></dd></div>" for a, b, c in stats)

    figs = []
    inline = bool(cfg.get("report", {}).get("inline_plotlyjs", False))
    figs.append(_fig_html(chart_cumulative(monthly), True, inline))
    figs.append(_fig_html(chart_deciles(dec, 12), False))
    figs.append(_fig_html(chart_ic(ic_ts), False))
    cross, date, bins = pe_scatter_data(panel)
    figs.append(_fig_html(chart_pe_scatter(cross, date, bins), False) if date is not None else "<p>Not enough data.</p>")
    figs.append(_fig_html(chart_corr(panel[panel["composite"].notna()]), False))
    write_pngs(out, monthly, dec, cross, date, bins, synthetic)

    # Strategy comparison table
    comp_rows = summary[summary["series"] == "strategy_net"].copy()
    comp_rows["Strategy"] = comp_rows["strategy"].map(lambda s: STRATEGY_NAMES.get(s, s))
    comp_rows["Rebalance"] = comp_rows["holding"].map(lambda h: "Monthly" if h == 1 else f"Every {h} months")
    strat_tbl = comp_rows[["Strategy", "Rebalance", "start", "end", "avg_holdings", "cagr", "sharpe",
                           "excess_return_ann", "t_stat_excess", "max_drawdown", "avg_turnover_per_rebalance"]]
    strat_tbl.columns = ["Strategy", "Rebalance", "From", "To", "Names", "CAGR", "Sharpe", "Excess vs universe",
                         "NW t-stat", "Max drawdown", "Turnover"]
    strat_html = _table(strat_tbl, {"Names": lambda v: num(v, 0), "CAGR": pct, "Sharpe": num,
                                    "Excess vs universe": lambda v: pct(v, signed=True), "NW t-stat": num,
                                    "Max drawdown": pct, "Turnover": lambda v: pct(v, 0)})

    ic_tbl = ic_sum.copy()
    ic_tbl["Score"] = ic_tbl["score"].map(lambda s: STRATEGY_NAMES.get(s, s))
    ic_piv = ic_tbl.pivot_table(index="Score", columns="horizon_months", values=["mean_ic", "t_stat"], sort=False)
    ic_piv.columns = [f"{'IC' if a == 'mean_ic' else 't'} {b}m" for a, b in ic_piv.columns]
    ic_piv = ic_piv[[c for h in sorted(ic_tbl["horizon_months"].unique()) for c in (f"IC {h}m", f"t {h}m")]].reset_index()
    ic_html = _table(ic_piv, {c: (lambda v: num(v, 3)) if c.startswith("IC") else num for c in ic_piv.columns if c != "Score"})

    screen_cols = {"rank": "#", "ticker": "Ticker", "name": "Company", "sector": "Sector", "label": "Label",
                   "composite": "Score", "pe": "P/E", "pb": "P/B", "ev_ebitda": "EV/EBITDA", "ev_ebit": "EV/EBIT",
                   "peg": "PEG", "momentum_12_1": "Mom 12-1", "market_cap_bn": "Mkt cap $bn"}
    fmt = {"Score": num, "P/E": lambda v: num(v, 1), "P/B": lambda v: num(v, 1), "EV/EBITDA": lambda v: num(v, 1),
           "EV/EBIT": lambda v: num(v, 1), "PEG": num, "Mom 12-1": lambda v: pct(v, 0, signed=True),
           "Mkt cap $bn": lambda v: num(v, 1), "#": lambda v: str(int(v))}
    top = screen.head(25)[list(screen_cols)].rename(columns=screen_cols)
    top_html = _table(top, {**fmt, "Label": _label_pill})
    rule_top = rule.head(25)[list(screen_cols)].rename(columns={**screen_cols, "composite": "Composite"})
    rule_html = _table(rule_top, {**fmt, "Composite": num, "Label": _label_pill})
    counts = screen["label"].value_counts()

    if model_status.get("status") == "ok":
        cv = pd.read_csv(out / "model_cv.csv")
        coefs = pd.read_csv(out / "model_coefficients.csv")
        cv_html = _table(cv[["fold", "train_start", "train_end", "test_start", "test_end", "auc_logit", "auc_composite",
                             "ic_logit", "ic_ridge", "ic_composite"]].rename(columns={
            "fold": "Fold", "train_start": "Train from", "train_end": "Train to", "test_start": "Test from",
            "test_end": "Test to", "auc_logit": "AUC logit", "auc_composite": "AUC composite", "ic_logit": "IC logit",
            "ic_ridge": "IC ridge", "ic_composite": "IC composite"}),
            {k: (lambda v: num(v, 3)) for k in ("AUC logit", "AUC composite", "IC logit", "IC ridge", "IC composite")})
        coefs["feature"] = coefs["feature"].map(lambda f: STRATEGY_NAMES.get(f, f))
        coef_html = _table(coefs.rename(columns={"feature": "Feature", "logit_coef_standardized": "Coefficient"}),
                           {"Coefficient": lambda v: num(v, 3)}, cls="narrow")
        model_html = (f"<p class='lede'>Folds split by month with a {cfg['model']['gap_months']}-month gap, so no training "
                      f"label overlaps a test month. AUC 0.5 is a coin flip. The model only earns its complexity if it "
                      f"beats the fixed-weight composite out of sample.</p>{cv_html}"
                      f"<h3>Coefficients on standardized factors (full-sample fit, for interpretation only)</h3>{coef_html}")
    else:
        model_html = f"<p class='lede'>{html.escape(model_status.get('message', 'Model stage not run.'))}</p>"

    qc_html = ""
    if len(qc):
        q = qc[["name", "status", "detail"]].copy()
        q["status"] = q["status"].map(lambda s: f'<span class="{s}">{s}</span>')
        q.columns = ["Check", "Status", "Detail"]
        qc_html = _table(q, {"Status": lambda v: v}, cls="left")

    banner = ('<div class="synthetic">Synthetic demo data. These numbers come from a simulated market with a planted '
              'value effect. They test the pipeline and say nothing about real stocks.</div>') if synthetic else ""
    est_share = run.get("filing_dates_estimated_share")
    sv = run.get("survivorship", {})
    if sv.get("membership_data"):
        surv = (f"Stocks are scored only in months they were index members, using membership rebuilt from the "
                f"dated change history (recorded back to {sv['earliest_recorded_change']}; earlier membership is "
                f"assumed). So the screen cannot buy a company before it joined. But {sv['removed_since_first_price']} "
                f"companies left the index during the test period and {sv['removed_missing_from_data']} of them have "
                f"no price data, so their returns, often poor, are missing: survivorship bias remains.")
    else:
        surv = ("This is a current-universe historical study: today's index members applied to the past. Companies "
                "that were dropped, acquired or went bankrupt are missing, which flatters historical returns "
                "(survivorship bias).")
    caveats = [
        surv,
        f"Fundamentals come from {html.escape(run['fundamentals_source'])}. "
        + (f"{pct(est_share, 0)} of statements have no filing date, so they are assumed public "
           f"{cfg['fundamentals']['filing_lag_days']} days after fiscal year-end." if est_share
           else "Every statement carries its filing date, so each is used only from the day it was published."),
        "Free sources give limited history, so the backtest window is only as long as the statement history allows. "
        "Short windows make Sharpe ratios and t-statistics noisy.",
        f"Costs are a flat {cfg['backtest']['cost_bps']} bps per unit of traded weight. No market impact, borrow or taxes.",
        (f"{pct(net['missing_return_share'], 1)} of holding-months had no return and were treated as cash. Treating "
         f"them as a total loss instead gives {pct(miss_row.iloc[0]['cagr'])} a year."
         if len(miss_row) and pd.notna(net.get("missing_return_share")) else "Missing-return sensitivity not available."),
        (f"EV-based factors need both debt and cash. In the latest month {int(ev_cov.iloc[-1]['ev_inputs_complete'])} of "
         f"{int(ev_cov.iloc[-1]['with_fundamentals'])} non-financial stocks had both; the rest were scored without "
         f"EV factors rather than assuming zero debt or cash." if len(ev_cov) else ""),
        "This is a research screen, not investment advice. Labels mark where a stock sits in the score distribution.",
    ]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Equity screen report, {run['as_of']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>{banner}<main>
<header class="top"><span><strong>Equity screen: value and momentum</strong></span>
<span class="meta"><span>Screen as of <strong>{run['as_of']}</strong></span><span>{run['n_universe']} stocks</span><span>Fundamentals from {html.escape(run['fundamentals_source'])}</span><span>Generated {run['generated_at_utc'][:10]}</span></span></header>
<p class="verdict">{verdict}</p>
<dl class="stats">{stats_html}</dl>

<section><h2>Retrospective backtest</h2>
<p class="lede">Illustrative: a retrospective research diagnostic, not an investable track record, and biased upward by survivorship (see the notes below). Each month the screen takes the top decile by composite score (at least {cfg['backtest']['min_holdings']} names) in equal weights and holds it for a month, {"trading one month after the score is computed" if head_lag else "trading at the same month-end close the score was computed from (an optimistic convention)"}. {"The same-close row shows the optimistic convention for comparison." if alt_lag == 0 else "The one-month-delay row is the conservative check."} Returns are total returns from adjusted prices; t-statistics use Newey-West standard errors because monthly returns are serially correlated.</p>
{figs[0]}
<h3>All strategies tested</h3>{strat_html}</section>

<section><h2>Does the score sort future returns?</h2>
<p class="lede">If the factors carry information, higher deciles should earn more and the rank information coefficient (IC) should sit above zero. Small ICs still matter when they are consistently positive; the t-statistic and the share of positive months show that consistency.</p>
<div class="grid2"><div>{figs[1]}</div><div>{figs[2]}</div></div>
<h3>Mean rank IC by forward horizon</h3>{ic_html}</section>

<section><h2>Valuation and forward returns</h2>
<p class="lede">Left: trailing P/E against the next 12 months' return for one cross-section, with the pooled median line across all months. Right: average cross-sectional rank correlation between factors, which shows how much each one adds.</p>
<div class="grid2"><div>{figs[3]}</div><div>{figs[4]}</div></div></section>

<section><h2>Current screen</h2>
<p class="lede">{counts.get('Research candidate', 0)} research candidates (top decile), {counts.get('Neutral', 0)} neutral and {counts.get('Lower priority', 0)} lower priority (bottom decile) out of {len(screen)} scored stocks. These are mechanical screening labels that decide where to look first, not recommendations. Full lists are in <code>screen_composite_latest.csv</code> and <code>screen_rule_based_latest.csv</code>.</p>
<h3>Composite score, top 25</h3>{top_html}
<h3>Rule-based baseline: lowest EV/EBIT with positive momentum, top 25</h3>{rule_html}</section>

<section><h2>Model check</h2>{model_html}</section>

<section><h2>Data and method notes</h2>
<ul class="notes">{''.join(f'<li>{c}</li>' for c in caveats)}</ul>
<h3>Data quality checks this run</h3>{qc_html}</section>
<footer>Run {html.escape(run['run_id'])}. Factor definitions and method: docs/factors.md and docs/methodology.md.</footer>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
