"""Discounted cash flow (FCFF) valuation for a single company.

    python -m eqscreen.dcf --ticker AAPL --init     # writes config/dcf/AAPL.yaml with history filled in
    #  ... edit the forward-looking assumptions in that file ...
    python -m eqscreen.dcf --ticker AAPL            # writes outputs/dcf/AAPL_<month>.md and CSVs

Data-derived (from the pipeline, point in time): base-year revenue, EBIT, D&A, capex,
debt, cash, diluted shares, share price and a 60-month beta.
Assumption-driven (set by you, never defaulted): revenue growth, EBIT margin, tax rate,
reinvestment ratios, terminal growth, risk-free rate, equity risk premium, cost of debt.
The model refuses to run while any of those is blank.

Method: unlevered free cash flow FCFF = EBIT x (1 - tax) + D&A - capex - change in NWC,
discounted at WACC with end-of-year timing; Gordon-growth terminal value; EV minus
debt plus cash gives equity value per diluted share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ROOT, load_config
from .db import connect, table_df
from .factors import load_panel, price_panel

REQUIRED = ["revenue_growth", "ebit_margin", "tax_rate", "da_pct_revenue", "capex_pct_revenue",
            "nwc_pct_incremental_revenue", "terminal_growth", "risk_free_rate", "equity_risk_premium",
            "pre_tax_cost_of_debt"]
SOURCE_KEYS = ["revenue_growth", "ebit_margin", "tax_rate", "reinvestment", "terminal_growth", "risk_free_rate",
               "equity_risk_premium", "pre_tax_cost_of_debt", "scenarios"]
MIN_SPREAD = 0.005          # WACC must exceed terminal growth by at least this much


class AssumptionError(ValueError):
    pass


def parse_date(value) -> pd.Timestamp | None:
    """A real date, or None when the value is absent. Raises ValueError for anything that is not
    a date, including values pandas parses to NaT ('NaT', 'nan')."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and not value.strip()):
        return None
    ts = pd.Timestamp(str(value))            # raises ValueError for text that is not a date
    if pd.isna(ts):
        raise ValueError(f"{value!r} is not a valid date")
    return ts


def month_end(value) -> pd.Timestamp | None:
    """'2016-06', '2016-06-15' or a Timestamp -> that month's last day; None stays None."""
    ts = parse_date(value)
    return None if ts is None else ts + pd.offsets.MonthEnd(0)


def retrospective(written_on, as_of: pd.Timestamp, tolerance_days: int = 45) -> bool:
    """True when something dated `written_on` was produced well after the `as_of` month-end.

    A report for a month-end is normally written in the following weeks, so a short tolerance
    (research.contemporaneous_days) separates contemporaneous work from hindsight.
    """
    if written_on is None:
        return False
    return pd.Timestamp(str(written_on)) > as_of + pd.Timedelta(days=int(tolerance_days))


def check_identity(a: dict, ticker: str, as_of=None) -> pd.Timestamp:
    """Refuse an assumptions file without identity metadata, or for another company or month.

    `ticker` and `as_of` are mandatory: without them nothing says what is valued, or when.
    `prepared_on`, if present, must be a date (missing means "undated", labelled in reports).
    Returns the valuation month.
    """
    missing = [k for k in ("ticker", "as_of") if not a.get(k)]
    if missing:
        raise AssumptionError(f"the assumptions file has no {' or '.join(missing)}; add it (ticker: XYZ, as_of: YYYY-MM) "
                              "or regenerate the file with --init")
    file_ticker = str(a["ticker"]).upper()
    if file_ticker != ticker.upper():
        raise AssumptionError(f"the assumptions file is for {file_ticker}, not {ticker.upper()}")
    try:
        file_month = month_end(a["as_of"])
        if file_month is None:
            raise ValueError("as_of is empty")
        parse_date(a.get("prepared_on"))   # absent is allowed (labelled undated); invalid is not
    except (ValueError, TypeError) as exc:
        raise AssumptionError(f"as_of / prepared_on must be dates (YYYY-MM / YYYY-MM-DD): {exc}") from exc
    requested = month_end(as_of)
    if requested is not None and file_month != requested:
        raise AssumptionError(f"the assumptions file is dated {file_month:%Y-%m} but the valuation was requested as "
                              f"of {requested:%Y-%m}; a DCF must use assumptions written for its own date. Create "
                              f"a file for {requested:%Y-%m} with --init --as-of {requested:%Y-%m}")
    return file_month


# --------------------------------------------------------------------------- data
def load_base(cfg: dict, ticker: str, as_of: str | None = None) -> dict:
    """Base-year inputs exactly as the screen saw them in one month (point in time)."""
    conn = connect(cfg["project"]["db_path"])
    t = ticker.upper()
    panel = load_panel(conn)
    rows = panel[(panel["ticker"] == t) & panel["period_end"].notna()]
    if as_of:
        rows = rows[rows["month_end"] <= pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)]
    if rows.empty:
        conn.close()
        raise AssumptionError(f"No statement-backed month for {t}{' on or before ' + as_of if as_of else ''}")
    row = rows.sort_values("month_end").iloc[-1]
    stmt = table_df(conn, "SELECT * FROM fundamentals WHERE ticker = ? AND source = ? AND period_end = ? "
                          "AND available_date = ?", (t, cfg["fundamentals"]["source"],
                                                     row["period_end"].strftime("%Y-%m-%d"),
                                                     row["available_date"].strftime("%Y-%m-%d"))).iloc[0]
    px = price_panel(conn)
    conn.close()
    bench = cfg["prices"]["benchmark"]
    r = px[t].pct_change(fill_method=None).loc[:row["month_end"]].dropna().tail(60)
    rb = px[bench].pct_change(fill_method=None).reindex(r.index) if bench in px else pd.Series(dtype=float)
    ok = r.notna() & rb.notna()
    beta = float(r[ok].cov(rb[ok]) / rb[ok].var()) if ok.sum() >= 24 and rb[ok].var() > 0 else np.nan
    return {
        "ticker": t, "name": row["name"], "sector": row["sector"], "as_of": row["month_end"],
        "period_end": row["period_end"], "available_date": row["available_date"],
        "version_type": stmt["version_type"], "accession": stmt["accession"],
        "revenue": stmt["revenue"], "ebit": stmt["ebit"],
        "da": stmt["ebitda"] - stmt["ebit"] if pd.notna(stmt["ebitda"]) and pd.notna(stmt["ebit"]) else np.nan,
        "capex": stmt["capex"], "operating_cash_flow": stmt["operating_cash_flow"],
        "total_debt": stmt["total_debt"], "cash": stmt["cash"],
        "shares": row["shares_adj"] if "shares_adj" in row else row["market_cap"] / row["close"],
        "price": row["close"], "market_cap": row["market_cap"],
        "beta_60m": beta, "beta_months": int(ok.sum()),
    }


def historical_reference(cfg: dict, ticker: str, as_of: pd.Timestamp, years: int = 4) -> dict:
    """Ratios from the last few fiscal years known at `as_of` (latest version of each year)."""
    conn = connect(cfg["project"]["db_path"])
    f = table_df(conn, "SELECT * FROM fundamentals WHERE ticker = ? AND source = ? AND available_date <= ? "
                       "ORDER BY period_end, available_date",
                 (ticker.upper(), cfg["fundamentals"]["source"], as_of.strftime("%Y-%m-%d")))
    conn.close()
    f = f.groupby("period_end").tail(1).tail(years + 1).reset_index(drop=True)
    rev = f["revenue"]

    def r(x):
        return None if x is None or pd.isna(x) else round(float(x), 4)

    def ratio(num, den):
        vals = [r(n / d) for n, d in zip(num, den) if pd.notna(n) and pd.notna(d) and d]
        return vals[-years:]
    return {
        "fiscal_years": f["period_end"].tolist()[-years:],
        "revenue_growth": [r(x) for x in (rev / rev.shift(1) - 1).tolist()[1:]][-years:],
        "ebit_margin": ratio(f["ebit"], rev),
        "da_pct_revenue": ratio(f["ebitda"] - f["ebit"], rev),
        "capex_pct_revenue": ratio(f["capex"], rev),
        "operating_cash_flow_to_net_income": ratio(f["operating_cash_flow"], f["net_income"]),
    }


# --------------------------------------------------------------------------- assumptions
def init_assumptions(cfg: dict, ticker: str, as_of: str | None, path: Path) -> Path:
    base = load_base(cfg, ticker, as_of)
    ref = historical_reference(cfg, ticker, base["as_of"])
    doc = {
        "ticker": base["ticker"],
        "as_of": base["as_of"].strftime("%Y-%m"),
        "prepared_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "_prepared_note": ("Date these assumptions were written. If it is well after as_of, reports label the "
                           "valuation retrospective (written with hindsight)."),
        "_note": ("Forward-looking assumptions are YOUR judgments and must be filled in (decimals, 0.05 = 5%). "
                  "The 'reference' block is history from the company's filings, shown for context only."),
        "revenue_growth": [None] * 5,
        "ebit_margin": None,
        "tax_rate": None,
        "da_pct_revenue": None,
        "capex_pct_revenue": None,
        "nwc_pct_incremental_revenue": None,
        "terminal_growth": None,
        "risk_free_rate": None,
        "equity_risk_premium": None,
        "pre_tax_cost_of_debt": None,
        "beta": None,
        "_beta_note": (f"Leave null to use the {base['beta_months']}-month beta from prices "
                       f"({base['beta_60m']:.2f})" if pd.notna(base["beta_60m"]) else
                       "No price-based beta available: set one"),
        "sensitivity": {"wacc_step": 0.005, "growth_step": 0.005, "steps": 2},
        "_scenarios_note": ("Bear and bull cases override any of the assumptions above; the top level is the base "
                            "case. Fill in or delete each scenario. scenario_weights is optional (must sum to 1)."),
        "scenarios": {
            "bear": {"revenue_growth": [None] * 5, "ebit_margin": None, "terminal_growth": None},
            "bull": {"revenue_growth": [None] * 5, "ebit_margin": None, "terminal_growth": None},
        },
        "scenario_weights": None,
        "_sources_note": ("Record where each input comes from and when (filing, data series and date, or your "
                          "rationale). Reports list these; blank sources are flagged."),
        "sources": {k: None for k in SOURCE_KEYS},
        "reference": ref,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and bool(np.isfinite(x))


MAX_SENSITIVITY_STEPS = 10      # 21 x 21 grid at most
MAX_SENSITIVITY_STEP = 0.05      # 5 percentage points between grid lines at most


def validate_sensitivity(a: dict) -> list[str]:
    """The optional `sensitivity` block: positive, finite step sizes and a small whole number of steps."""
    s = a.get("sensitivity")
    if s is None:
        return []
    if not isinstance(s, dict):
        return ["sensitivity must be a mapping with wacc_step, growth_step and steps"]
    problems = []
    for key in ("wacc_step", "growth_step"):
        if key in s and not (_finite(s[key]) and 0 < s[key] <= MAX_SENSITIVITY_STEP):
            problems.append(f"sensitivity.{key} must be a finite number above 0 and at most "
                            f"{MAX_SENSITIVITY_STEP} (got {s[key]!r})")
    if "steps" in s and not (isinstance(s["steps"], (int, np.integer)) and not isinstance(s["steps"], bool)
                             and 0 <= s["steps"] <= MAX_SENSITIVITY_STEPS):
        problems.append(f"sensitivity.steps must be a whole number from 0 to {MAX_SENSITIVITY_STEPS} "
                        f"(got {s['steps']!r})")
    return problems


def validate(a: dict) -> list[str]:
    problems = []
    for key in REQUIRED:
        v = a.get(key)
        if v is None or (isinstance(v, list) and (not v or any(x is None for x in v))):
            problems.append(f"{key} is not set")
    if problems:
        return problems
    # Every number must be a finite real number: NaN or text would silently poison the valuation.
    for key in REQUIRED:
        vals = a[key] if isinstance(a[key], list) else [a[key]]
        if not all(_finite(x) for x in vals):
            problems.append(f"{key} must contain finite numbers (got {a[key]!r})")
    for key in ("beta", "total_debt", "cash"):
        if a.get(key) is not None and not _finite(a[key]):
            problems.append(f"{key} must be a finite number when set (got {a[key]!r})")
    for key in ("total_debt", "cash"):
        if _finite(a.get(key)) and a[key] < 0:
            problems.append(f"{key} cannot be negative")
    problems += validate_sensitivity(a)
    if problems:
        return problems
    n = len(a["revenue_growth"]) if isinstance(a["revenue_growth"], list) else 0
    if n < 1:
        problems.append("revenue_growth must be a list with one rate per forecast year")
    if isinstance(a["ebit_margin"], list) and len(a["ebit_margin"]) != n:
        problems.append("ebit_margin must be one number or a list as long as revenue_growth")
    if not 0 <= a["tax_rate"] < 1:
        problems.append("tax_rate must be between 0 and 1")
    if isinstance(a["revenue_growth"], list) and any(g <= -1 or g > 3 for g in a["revenue_growth"]):
        problems.append("revenue_growth values must be above -100% and at most 300% a year")
    margins = a["ebit_margin"] if isinstance(a["ebit_margin"], list) else [a["ebit_margin"]]
    if any(not -1 < mg < 1 for mg in margins):
        problems.append("ebit_margin must be between -100% and 100% (decimals, e.g. 0.18)")
    for key in ("da_pct_revenue", "capex_pct_revenue", "nwc_pct_incremental_revenue"):
        if not 0 <= a[key] <= 1:
            problems.append(f"{key} must be between 0 and 1")
    if not -0.05 <= a["terminal_growth"] < 0.5:
        problems.append("terminal_growth looks wrong; use decimals, e.g. 0.025")
    for key in ("risk_free_rate", "equity_risk_premium", "pre_tax_cost_of_debt"):
        if not 0 <= a[key] < 0.5:
            problems.append(f"{key} looks wrong ({a[key]}); use decimals, e.g. 0.04")
    return problems


NON_ASSUMPTION_KEYS = {"scenarios", "scenario_weights", "reference", "sensitivity", "ticker", "as_of"}


def scenario_assumptions(a: dict) -> dict[str, dict]:
    """{"base": a, "bear": a + overrides, ...}; scenario blocks may override any assumption."""
    base = {k: v for k, v in a.items() if k not in ("scenarios", "scenario_weights")}
    out = {"base": base}
    for name, overrides in (a.get("scenarios") or {}).items():
        out[name] = {**base, **(overrides or {})}
    return out


def validate_scenarios(a: dict) -> list[str]:
    problems = []
    for name, s in scenario_assumptions(a).items():
        if name == "base":
            continue
        for p in validate(s):
            problems.append(f"scenario '{name}': {p} (fill it in or delete the scenario)")
    weights = a.get("scenario_weights")
    if weights:
        names = set(scenario_assumptions(a))
        if set(weights) != names:
            problems.append(f"scenario_weights must cover exactly {sorted(names)}")
        else:
            bad = [k for k, w in weights.items()
                   if not isinstance(w, (int, float)) or isinstance(w, bool) or not np.isfinite(w) or not 0 <= w <= 1]
            if bad:
                problems.append(f"scenario_weights must be probabilities between 0 and 1 (check {', '.join(sorted(bad))})")
            elif abs(sum(weights.values()) - 1) > 1e-6:
                problems.append("scenario_weights must sum to 1")
    return problems


def value_scenarios(base: dict, a: dict) -> pd.DataFrame:
    problems = validate(a) + validate_scenarios(a)
    if problems:
        raise AssumptionError("Fill in the assumptions file first: " + "; ".join(problems))
    rows = []
    weights = a.get("scenario_weights") or {}
    for name, s in scenario_assumptions(a).items():
        r = value(base, s)
        g = s["revenue_growth"]
        m = s["ebit_margin"]
        rows.append({"scenario": name, "avg_revenue_growth": float(np.mean(g)),
                     "ebit_margin_final": float(m[-1] if isinstance(m, list) else m),
                     "terminal_growth": s["terminal_growth"], "wacc": r["wacc"], "enterprise_value": r["ev"],
                     "value_per_share": r["value_per_share"], "upside": r["upside"],
                     "terminal_share": r["terminal_share"], "weight": weights.get(name)})
    order = {"bear": 0, "base": 1, "bull": 2}
    return pd.DataFrame(rows).sort_values("scenario", key=lambda s: s.map(lambda x: order.get(x, 3))).reset_index(drop=True)


def weighted_value(scen: pd.DataFrame) -> float | None:
    """Probability-weighted value; None without weights. Refuses if any scenario value is invalid
    (pandas' sum would otherwise skip it and return a plausible-looking but wrong number)."""
    if scen["weight"].isna().any():
        return None
    values = scen["value_per_share"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssumptionError("a scenario has no valid value per share; refusing to compute a weighted value")
    return float(np.dot(scen["weight"].to_numpy(dtype=float), values))


def _path(a: dict, key: str, n: int) -> list[float]:
    v = a[key]
    return [float(x) for x in v] if isinstance(v, list) else [float(v)] * n


# --------------------------------------------------------------------------- model
def wacc_build(base: dict, a: dict) -> dict:
    beta = a.get("beta")
    if beta is None:
        beta = base["beta_60m"]
        if beta is None or pd.isna(beta):
            raise AssumptionError("No price-based beta could be computed; set `beta` in the assumptions file")
    debt = a.get("total_debt", base["total_debt"])
    equity = base["market_cap"]
    if debt is None or pd.isna(debt) or equity is None or pd.isna(equity) or equity <= 0:
        raise AssumptionError("Market cap and total debt are needed for WACC weights; debt is missing "
                              "(set `total_debt` in the assumptions file from the filing)")
    ke = a["risk_free_rate"] + beta * a["equity_risk_premium"]
    kd = a["pre_tax_cost_of_debt"] * (1 - a["tax_rate"])
    we = equity / (equity + debt)
    wacc = we * ke + (1 - we) * kd
    if not np.isfinite(wacc):
        raise AssumptionError(f"WACC is not a finite number (beta {beta}, debt {debt}, market cap {equity})")
    return {"beta": float(beta), "cost_of_equity": ke, "after_tax_cost_of_debt": kd, "equity_weight": we,
            "debt_weight": 1 - we, "wacc": wacc}


def project(revenue0: float, a: dict) -> pd.DataFrame:
    n = len(a["revenue_growth"])
    growth, margin = _path(a, "revenue_growth", n), _path(a, "ebit_margin", n)
    rows, prev = [], revenue0
    for t in range(1, n + 1):
        rev = prev * (1 + growth[t - 1])
        ebit = rev * margin[t - 1]
        nopat = ebit * (1 - a["tax_rate"])
        da = rev * a["da_pct_revenue"]
        capex = rev * a["capex_pct_revenue"]
        d_nwc = (rev - prev) * a["nwc_pct_incremental_revenue"]
        rows.append({"year": t, "revenue": rev, "growth": growth[t - 1], "ebit": ebit, "ebit_margin": margin[t - 1],
                     "nopat": nopat, "da": da, "capex": capex, "change_nwc": d_nwc, "fcff": nopat + da - capex - d_nwc})
        prev = rev
    return pd.DataFrame(rows)


def enterprise_value(proj: pd.DataFrame, wacc: float, g: float) -> dict:
    if wacc - g < MIN_SPREAD:
        raise AssumptionError(f"WACC ({wacc:.2%}) must exceed terminal growth ({g:.2%}) by at least "
                              f"{MIN_SPREAD:.1%}; the terminal value is otherwise undefined or explosive")
    disc = (1 + wacc) ** -proj["year"].to_numpy(dtype=float)
    pv = proj["fcff"].to_numpy() * disc
    tv = proj["fcff"].iloc[-1] * (1 + g) / (wacc - g)
    pv_tv = tv * disc[-1]
    return {"pv_fcff": pv, "terminal_value": tv, "pv_terminal": pv_tv, "ev": pv.sum() + pv_tv,
            "terminal_share": pv_tv / (pv.sum() + pv_tv)}


def value(base: dict, a: dict) -> dict:
    problems = validate(a)
    if problems:
        raise AssumptionError("Fill in the assumptions file first: " + "; ".join(problems))
    for key in ("revenue", "shares", "price"):
        if base[key] is None or pd.isna(base[key]):
            raise AssumptionError(f"Base-year {key} is missing for {base['ticker']}")
        if base[key] <= 0:
            raise AssumptionError(f"Base-year {key} is not positive for {base['ticker']} ({base[key]}); "
                                  "a DCF on this base would be meaningless")
    cash = a.get("cash", base["cash"])
    if cash is None or pd.isna(cash):
        raise AssumptionError("Cash is missing for the equity bridge (set `cash` in the assumptions file)")
    w = wacc_build(base, a)
    proj = project(base["revenue"], a)
    ev = enterprise_value(proj, w["wacc"], a["terminal_growth"])
    debt = a.get("total_debt", base["total_debt"])
    equity = ev["ev"] - debt + cash
    per_share = equity / base["shares"]
    if not np.isfinite(per_share):
        raise AssumptionError("the valuation produced a non-finite value per share; check the inputs")
    proj["discount_factor"] = (1 + w["wacc"]) ** -proj["year"]
    proj["pv_fcff"] = ev["pv_fcff"]
    return {**w, **ev, "projection": proj, "debt": debt, "cash": cash, "equity_value": equity,
            "value_per_share": per_share, "price": base["price"], "upside": per_share / base["price"] - 1}


def _bisect(f, lo: float, hi: float, tol: float = 1e-7, iters: int = 200) -> float | None:
    """Root of f on [lo, hi] if f changes sign there, else None."""
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def implied_assumptions(base: dict, a: dict) -> dict:
    """Reverse DCF: what the current share price implies, holding the other base assumptions.

    * terminal growth that makes value per share equal the price;
    * a single constant revenue growth rate for every forecast year that does the same.
    None when the price lies outside what the searched range can produce.
    """
    w = wacc_build(base, a)["wacc"]
    debt, cash = a.get("total_debt", base["total_debt"]), a.get("cash", base["cash"])
    price, shares = base["price"], base["shares"]

    def per_share(assume, g):
        try:
            ev = enterprise_value(project(base["revenue"], assume), w, g)["ev"]
        except AssumptionError:
            return np.nan
        return (ev - debt + cash) / shares

    g_hi = w - MIN_SPREAD - 1e-6
    implied_g = _bisect(lambda g: per_share(a, g) - price, -0.05, g_hi)
    n = len(a["revenue_growth"])
    implied_rev = _bisect(lambda r: per_share({**a, "revenue_growth": [r] * n}, a["terminal_growth"]) - price,
                          -0.5, 1.0)
    return {"wacc": w, "implied_terminal_growth": implied_g, "implied_revenue_growth": implied_rev,
            "search_terminal_growth": (-0.05, g_hi), "search_revenue_growth": (-0.5, 1.0)}


def sensitivity(base: dict, a: dict, wacc: float) -> pd.DataFrame:
    problems = validate_sensitivity(a)
    if problems:
        raise AssumptionError("; ".join(problems))
    s = a.get("sensitivity") or {}
    ws, gs, k = float(s.get("wacc_step", 0.005)), float(s.get("growth_step", 0.005)), int(s.get("steps", 2))
    proj = project(base["revenue"], a)
    debt, cash = a.get("total_debt", base["total_debt"]), a.get("cash", base["cash"])
    grid = {}
    for dg in range(-k, k + 1):
        g = a["terminal_growth"] + dg * gs
        col = {}
        for dw in range(-k, k + 1):
            w = wacc + dw * ws
            try:
                col[round(w, 6)] = (enterprise_value(proj, w, g)["ev"] - debt + cash) / base["shares"]
            except AssumptionError:
                col[round(w, 6)] = np.nan
        grid[round(g, 6)] = col
    out = pd.DataFrame(grid)
    out.index.name, out.columns.name = "wacc", "terminal_growth"
    return out


# --------------------------------------------------------------------------- output
def _p(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}%}"


def _m(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x / 1e9:,.2f} bn" if abs(x) >= 1e8 else f"{x / 1e6:,.1f} m"


def render_scenarios(scen: pd.DataFrame, price: float) -> list[str]:
    if len(scen) <= 1:
        return []
    lines = ["## Scenarios", "",
             "| Scenario | Avg revenue growth | Final EBIT margin | Terminal growth | WACC | Value per share | vs price "
             "| Terminal share | Weight |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in scen.itertuples():
        lines.append(f"| {r.scenario.title()} | {_p(r.avg_revenue_growth)} | {_p(r.ebit_margin_final)} | "
                     f"{_p(r.terminal_growth)} | {_p(r.wacc, 2)} | {r.value_per_share:,.2f} | {r.upside:+.1%} | "
                     f"{_p(r.terminal_share, 0)} | {'n/a' if pd.isna(r.weight) else _p(r.weight, 0)} |")
    wv = weighted_value(scen)
    lines += ["", f"Range: {scen['value_per_share'].min():,.2f} to {scen['value_per_share'].max():,.2f} per share "
                  f"against a price of {price:,.2f}."
              + (f" Probability-weighted value: {wv:,.2f} ({wv / price - 1:+.1%})." if wv is not None else ""), ""]
    return lines


def render_implied(imp: dict, a: dict) -> list[str]:
    g, r = imp["implied_terminal_growth"], imp["implied_revenue_growth"]
    lines = ["## What the share price implies (reverse DCF)", "",
             "Holding every other base-case assumption fixed, the current price is consistent with:", "",
             "| Solved for | Implied by the price | Your base case |", "|---|---:|---:|",
             f"| Terminal growth | {_p(g, 2) if g is not None else 'outside ' + _p(imp['search_terminal_growth'][0]) + ' to ' + _p(imp['search_terminal_growth'][1], 2)} "
             f"| {_p(a['terminal_growth'], 2)} |",
             f"| Constant revenue growth, every forecast year | {_p(r, 1) if r is not None else 'outside -50% to 100%'} "
             f"| {', '.join(_p(x) for x in a['revenue_growth'])} |", ""]
    if g is not None and g > a["risk_free_rate"]:
        lines += [f"The implied terminal growth exceeds your risk-free rate ({_p(a['risk_free_rate'], 2)}), a common "
                  "sign that the market is pricing in more than steady-state growth.", ""]
    return lines


def missing_sources(a: dict) -> list[str]:
    src = a.get("sources") or {}
    keys = [k for k in SOURCE_KEYS if k != "scenarios" or a.get("scenarios")]
    return [k for k in keys if not src.get(k)]


def render_sources(a: dict) -> list[str]:
    src = a.get("sources") or {}
    keys = [k for k in SOURCE_KEYS if k != "scenarios" or a.get("scenarios")]
    lines = ["## Sources of the assumptions", "", "| Input | Source and date |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ').capitalize()} | {src.get(k) or '**not recorded**'} |" for k in keys]
    miss = missing_sources(a)
    if miss:
        lines += ["", f"{len(miss)} input(s) have no recorded source. Add them under `sources:` in the assumptions "
                      "file before sharing this valuation."]
    return lines + [""]


def render(cfg: dict, base: dict, a: dict, res: dict, sens: pd.DataFrame,
           scen: pd.DataFrame | None = None, imp: dict | None = None) -> str:
    proj = res["projection"]
    ref = a.get("reference", {})
    lines = []
    if cfg["mode"] == "demo":
        lines += ["> **SYNTHETIC DEMO DATA.** Generated from a simulated market to check the model. "
                  "It values no real company.", ""]
    tol = int(cfg.get("research", {}).get("contemporaneous_days", 45))
    if not a.get("prepared_on"):
        lines += ["> **Undated assumptions.** The file records no `prepared_on` date, so it cannot show whether "
                  "these assumptions were written at the valuation date or with hindsight.", ""]
    elif retrospective(a.get("prepared_on"), base["as_of"], tol):
        lines += [f"> **Retrospective valuation.** These assumptions were written on {a['prepared_on']}, after the "
                  f"{base['as_of']:%Y-%m-%d} valuation date, and may reflect information not available then.", ""]
    lines += [f"# {base['name'] or base['ticker']} ({base['ticker']}): DCF valuation", "",
              f"Valuation month-end: **{base['as_of']:%Y-%m-%d}**. Base year: fiscal year to "
              f"{base['period_end']:%Y-%m-%d} ({base['version_type']}, public from {base['available_date']:%Y-%m-%d}).",
              "",
              f"**Equity value {res['value_per_share']:,.2f} per diluted share against a price of {base['price']:,.2f}"
              f" ({res['upside']:+.1%}).** The terminal value is {res['terminal_share']:.0%} of enterprise value, so "
              "the result depends heavily on the long-run assumptions; see the sensitivity table.", "",
              *([f"Note: terminal growth ({_p(a['terminal_growth'], 2)}) is above the risk-free rate "
                 f"({_p(a['risk_free_rate'], 2)}). Long-run growth above the risk-free rate is hard to justify for "
                 "most companies; consider lowering it.", ""] if a["terminal_growth"] > a["risk_free_rate"] else []),
              "## Assumptions", "",
              "| Assumption | Used | Recent history (from filings) |", "|---|---:|---:|"]
    hist = lambda k: ", ".join(_p(x) for x in ref.get(k, []) if x is not None) or "n/a"  # noqa: E731
    g = a["revenue_growth"]
    m = a["ebit_margin"]
    lines += [f"| Revenue growth, years 1-{len(g)} | {', '.join(_p(x) for x in g)} | {hist('revenue_growth')} |",
              f"| EBIT margin | {', '.join(_p(x) for x in m) if isinstance(m, list) else _p(m)} | {hist('ebit_margin')} |",
              f"| D&A, % of revenue | {_p(a['da_pct_revenue'])} | {hist('da_pct_revenue')} |",
              f"| Capex, % of revenue | {_p(a['capex_pct_revenue'])} | {hist('capex_pct_revenue')} |",
              f"| Net working capital, % of revenue growth | {_p(a['nwc_pct_incremental_revenue'])} | n/a |",
              f"| Tax rate | {_p(a['tax_rate'])} | n/a |",
              f"| Terminal growth | {_p(a['terminal_growth'])} | n/a |", "",
              "## Discount rate (WACC)", "",
              "| Component | Value |", "|---|---:|",
              f"| Risk-free rate (your input) | {_p(a['risk_free_rate'], 2)} |",
              f"| Equity risk premium (your input) | {_p(a['equity_risk_premium'], 2)} |",
              f"| Beta | {res['beta']:.2f} " + ("(your input)" if a.get("beta") is not None else
                                                f"({base['beta_months']}-month regression on {cfg['prices']['benchmark']})")
              + " |",
              f"| Cost of equity | {_p(res['cost_of_equity'], 2)} |",
              f"| After-tax cost of debt | {_p(res['after_tax_cost_of_debt'], 2)} |",
              f"| Weights, equity / debt (market cap / book debt) | {_p(res['equity_weight'], 0)} / {_p(res['debt_weight'], 0)} |",
              f"| **WACC** | **{_p(res['wacc'], 2)}** |", "",
              "## Free cash flow forecast", "",
              "| Year | Revenue | EBIT | NOPAT | + D&A | − Capex | − ΔNWC | FCFF | PV of FCFF |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.year} | {_m(r.revenue)} | {_m(r.ebit)} | {_m(r.nopat)} | {_m(r.da)} | {_m(r.capex)} | "
              f"{_m(r.change_nwc)} | {_m(r.fcff)} | {_m(r.pv_fcff)} |" for r in proj.itertuples()]
    lines += [f"| Base | {_m(base['revenue'])} | {_m(base['ebit'])} | | {_m(base['da'])} | {_m(base['capex'])} | | | |",
              "", "## From enterprise value to value per share", "",
              "| Item | Value |", "|---|---:|",
              f"| PV of forecast FCFF | {_m(res['pv_fcff'].sum())} |",
              f"| PV of terminal value | {_m(res['pv_terminal'])} |",
              f"| Enterprise value | {_m(res['ev'])} |",
              f"| − Total debt | {_m(res['debt'])} |",
              f"| + Cash | {_m(res['cash'])} |",
              f"| Equity value | {_m(res['equity_value'])} |",
              f"| Diluted shares (split-adjusted) | {base['shares'] / 1e6:,.1f} m |",
              f"| **Value per share** | **{res['value_per_share']:,.2f}** |", "",
              *(render_scenarios(scen, base["price"]) if scen is not None else []),
              *(render_implied(imp, a) if imp is not None else []),
              "## Sensitivity: value per share by WACC (rows) and terminal growth (columns)", "",
              "| WACC | " + " | ".join(_p(c) for c in sens.columns) + " |",
              "|---:|" + "---:|" * len(sens.columns)]
    for w, row in sens.iterrows():
        cells = ["n/a" if pd.isna(v) else f"{v:,.2f}" for v in row]
        lines.append(f"| {_p(w, 2)} | " + " | ".join(cells) + " |")
    lines += ["", "n/a: WACC too close to terminal growth for a meaningful terminal value.", ""]
    lines += render_sources(a)
    lines += [
              "## Method and limits", "",
              "- FCFF = EBIT × (1 − tax) + D&A − capex − change in net working capital; end-of-year discounting.",
              "- Terminal value by Gordon growth on year-N FCFF. Equity = EV − total debt + cash; no adjustment "
              "for leases, minority interests, pensions or options beyond diluted shares.",
              "- WACC weights use market equity and book debt. Beta is a raw regression on monthly returns unless set.",
              "- Base-year figures are the statement version public at the valuation date (point in time).",
              "- Forward-looking assumptions are the analyst's; history is shown for context, not used directly.", ""]
    return "\n".join(lines)


def run(cfg: dict, ticker: str, assumptions_path: Path, as_of: str | None = None) -> Path:
    a = yaml.safe_load(assumptions_path.read_text(encoding="utf-8"))
    month = check_identity(a, ticker, as_of)
    base = load_base(cfg, ticker, month.strftime("%Y-%m"))
    if base["as_of"] != month:
        raise AssumptionError(f"no statement-backed valuation base for {ticker.upper()} at {month:%Y-%m} "
                              f"(latest available is {base['as_of']:%Y-%m})")
    scen = value_scenarios(base, a)
    res = value(base, scenario_assumptions(a)["base"])
    sens = sensitivity(base, a, res["wacc"])
    imp = implied_assumptions(base, scenario_assumptions(a)["base"])
    out = Path(cfg["project"]["output_dir"]) / "dcf"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{base['ticker']}_{base['as_of']:%Y%m}"
    res["projection"].to_csv(out / f"{stem}_projection.csv", index=False, float_format="%.6g")
    sens.to_csv(out / f"{stem}_sensitivity.csv", float_format="%.6g")
    scen.to_csv(out / f"{stem}_scenarios.csv", index=False, float_format="%.6g")
    path = out / f"{stem}.md"
    path.write_text(render(cfg, base, a, res, sens, scen, imp), encoding="utf-8")
    from .render import write_html
    write_html(path, f"{base['ticker']} DCF valuation", "Generated by eqscreen. A valuation under stated assumptions, "
                                                       "not a price target.")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DCF valuation for one company")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--init", action="store_true", help="write an assumptions file with history filled in")
    ap.add_argument("--assumptions", default=None, help="default: config/dcf/<TICKER>.yaml")
    ap.add_argument("--as-of", default=None, help="YYYY-MM valuation month (default: from the file, else latest)")
    ap.add_argument("--mode", choices=["live", "demo"], default="live")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, mode=args.mode)
    path = Path(args.assumptions) if args.assumptions else ROOT / "config" / "dcf" / f"{args.ticker.upper()}.yaml"
    if args.init:
        if path.exists():
            raise SystemExit(f"{path} already exists; delete it or pass --assumptions to write elsewhere")
        print(f"Assumptions template written to {init_assumptions(cfg, args.ticker, args.as_of, path)}")
        print("Fill in every null forward-looking field, then run again without --init.")
        return
    try:
        print(f"DCF written to {run(cfg, args.ticker, path, args.as_of)}")
    except AssumptionError as exc:
        raise SystemExit(f"Cannot value {args.ticker.upper()}: {exc}")


if __name__ == "__main__":
    main()
