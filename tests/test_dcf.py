"""DCF engine tests. All numbers are made-up test inputs, not market data."""
import numpy as np
import pandas as pd
import pytest
import yaml

from eqscreen import dcf

BASE = {"ticker": "T", "name": "Test Co", "revenue": 1000.0, "ebit": 150.0, "da": 40.0, "capex": 50.0,
        "total_debt": 300.0, "cash": 100.0, "shares": 100.0, "price": 20.0, "market_cap": 2000.0,
        "beta_60m": 1.2, "beta_months": 60}
ASSUME = {"revenue_growth": [0.03] * 5, "ebit_margin": 0.15, "tax_rate": 0.25, "da_pct_revenue": 0.04,
          "capex_pct_revenue": 0.05, "nwc_pct_incremental_revenue": 0.10, "terminal_growth": 0.03,
          "risk_free_rate": 0.04, "equity_risk_premium": 0.05, "pre_tax_cost_of_debt": 0.06}


def test_constant_growth_dcf_equals_gordon_perpetuity():
    """With constant growth and ratios, FCFF grows at g from year 1, so EV = FCFF_1 / (WACC - g)."""
    proj = dcf.project(BASE["revenue"], ASSUME)
    ev = dcf.enterprise_value(proj, 0.09, 0.03)["ev"]
    assert ev == pytest.approx(proj["fcff"].iloc[0] / (0.09 - 0.03), rel=1e-12)


def test_projection_arithmetic_year_one():
    r = dcf.project(1000.0, ASSUME).iloc[0]
    assert r["revenue"] == pytest.approx(1030)
    assert r["nopat"] == pytest.approx(1030 * 0.15 * 0.75)
    assert r["fcff"] == pytest.approx(1030 * 0.15 * 0.75 + 1030 * 0.04 - 1030 * 0.05 - 30 * 0.10)


def test_wacc_build():
    w = dcf.wacc_build(BASE, ASSUME)
    ke, kd, we = 0.04 + 1.2 * 0.05, 0.06 * 0.75, 2000 / 2300
    assert w["cost_of_equity"] == pytest.approx(ke)
    assert w["wacc"] == pytest.approx(we * ke + (1 - we) * kd)
    assert dcf.wacc_build(BASE, {**ASSUME, "beta": 0.8})["beta"] == 0.8


def test_equity_bridge_and_per_share():
    res = dcf.value(BASE, ASSUME)
    assert res["equity_value"] == pytest.approx(res["ev"] - 300 + 100)
    assert res["value_per_share"] == pytest.approx(res["equity_value"] / 100)
    assert res["upside"] == pytest.approx(res["value_per_share"] / 20 - 1)
    assert 0 < res["terminal_share"] < 1


def test_blank_assumptions_are_refused_and_listed():
    blank = {k: None for k in ASSUME} | {"revenue_growth": [None] * 5}
    with pytest.raises(dcf.AssumptionError) as e:
        dcf.value(BASE, blank)
    for key in ("revenue_growth", "risk_free_rate", "terminal_growth"):
        assert key in str(e.value)


def test_terminal_growth_must_stay_below_wacc():
    with pytest.raises(dcf.AssumptionError):
        dcf.enterprise_value(dcf.project(1000.0, ASSUME), 0.05, 0.048)


def test_missing_debt_requires_explicit_override():
    with pytest.raises(dcf.AssumptionError):
        dcf.value({**BASE, "total_debt": np.nan}, ASSUME)
    assert dcf.value({**BASE, "total_debt": np.nan}, {**ASSUME, "total_debt": 300.0})["debt"] == 300.0


def test_sensitivity_moves_the_right_way():
    res = dcf.value(BASE, ASSUME)
    grid = dcf.sensitivity(BASE, ASSUME, res["wacc"])
    assert grid.shape == (5, 5)
    centre = grid.iloc[2, 2]
    assert centre == pytest.approx(res["value_per_share"])
    finite = grid.dropna(how="any")
    assert (finite.diff().dropna() < 0).all().all()               # higher WACC, lower value
    assert (finite.T.diff().dropna() > 0).all().all()             # higher growth, higher value


def test_demo_end_to_end_init_fill_and_value(demo_run, tmp_path):
    screen = pd.read_csv(f"{demo_run['project']['output_dir']}/screen_composite_latest.csv")
    tk = screen[screen["sector"] != "Financials"]["ticker"].iloc[0]
    path = dcf.init_assumptions(demo_run, tk, None, tmp_path / f"{tk}.yaml")
    doc = yaml.safe_load(path.read_text())
    assert doc["revenue_growth"] == [None] * 5 and doc["risk_free_rate"] is None
    assert doc["reference"]["ebit_margin"], "history from filings should be filled in"
    with pytest.raises(dcf.AssumptionError):
        dcf.run(demo_run, tk, path)                                 # refuses while blank
    doc.update({k: v for k, v in ASSUME.items()})
    doc["scenarios"] = {"bear": {"revenue_growth": [0.0] * 5, "ebit_margin": 0.12, "terminal_growth": 0.02},
                        "bull": {"revenue_growth": [0.06] * 5, "ebit_margin": 0.18, "terminal_growth": 0.03}}
    path.write_text(yaml.safe_dump(doc))
    out = dcf.run(demo_run, tk, path)
    text = out.read_text()
    assert text.startswith("> **SYNTHETIC DEMO DATA.**")
    for heading in ("Assumptions", "Discount rate (WACC)", "Free cash flow forecast",
                    "From enterprise value to value per share", "Scenarios", "Sensitivity"):
        assert f"## {heading}" in text
    assert (out.parent / out.name.replace(".md", "_sensitivity.csv")).exists()


def test_scenarios_override_base_and_order_by_value():
    a = {**ASSUME, "scenarios": {"bear": {"revenue_growth": [0.0] * 5, "ebit_margin": 0.12, "terminal_growth": 0.02},
                                 "bull": {"revenue_growth": [0.08] * 5, "ebit_margin": 0.18}},
         "scenario_weights": {"bear": 0.25, "base": 0.5, "bull": 0.25}}
    scen = dcf.value_scenarios(BASE, a).set_index("scenario")
    assert list(scen.index) == ["bear", "base", "bull"]
    assert scen.loc["bear", "value_per_share"] < scen.loc["base", "value_per_share"] < scen.loc["bull", "value_per_share"]
    assert scen.loc["base", "value_per_share"] == pytest.approx(dcf.value(BASE, ASSUME)["value_per_share"])
    assert scen.loc["bull", "terminal_growth"] == ASSUME["terminal_growth"]      # not overridden: inherits base
    assert dcf.weighted_value(scen.reset_index()) == pytest.approx(
        0.25 * scen.loc["bear", "value_per_share"] + 0.5 * scen.loc["base", "value_per_share"]
        + 0.25 * scen.loc["bull", "value_per_share"])


def test_incomplete_scenarios_and_bad_weights_are_refused():
    blank = {**ASSUME, "scenarios": {"bear": {"revenue_growth": [None] * 5, "ebit_margin": None}}}
    with pytest.raises(dcf.AssumptionError, match="scenario 'bear'"):
        dcf.value_scenarios(BASE, blank)
    bad = {**ASSUME, "scenarios": {"bear": {"ebit_margin": 0.1}}, "scenario_weights": {"bear": 0.5, "base": 0.6}}
    with pytest.raises(dcf.AssumptionError, match="sum to 1"):
        dcf.value_scenarios(BASE, bad)


# --------------------------------------------------------------------------- extreme inputs
@pytest.mark.parametrize("g", [0.09, 0.10, 0.15])          # equal to, and above, a 9% WACC
def test_growth_at_or_above_wacc_is_refused(g):
    with pytest.raises(dcf.AssumptionError, match="must exceed terminal growth"):
        dcf.enterprise_value(dcf.project(1000.0, ASSUME), 0.09, g)


def test_spread_just_above_minimum_is_accepted():
    ev = dcf.enterprise_value(dcf.project(1000.0, ASSUME), 0.09, 0.09 - dcf.MIN_SPREAD - 1e-4)["ev"]
    assert np.isfinite(ev) and ev > 0


def test_full_valuation_refuses_terminal_growth_above_wacc():
    with pytest.raises(dcf.AssumptionError):
        dcf.value(BASE, {**ASSUME, "terminal_growth": 0.30})


@pytest.mark.parametrize("growth", [[1.0] * 5, [-0.3] * 5, [0.0] * 5])
def test_extreme_forecast_growth_stays_finite(growth):
    res = dcf.value(BASE, {**ASSUME, "revenue_growth": growth})
    assert np.isfinite(res["value_per_share"])


def test_sensitivity_grid_blanks_cells_where_growth_reaches_wacc():
    a = {**ASSUME, "terminal_growth": 0.07}
    w = dcf.wacc_build(BASE, a)["wacc"]                  # about 9.6%
    grid = dcf.sensitivity(BASE, {**a, "sensitivity": {"wacc_step": 0.01, "growth_step": 0.01, "steps": 2}}, w)
    for wacc, row in grid.iterrows():
        for g, v in row.items():
            assert np.isnan(v) == (wacc - g < dcf.MIN_SPREAD)
    assert grid.isna().any().any() and grid.notna().any().any()


def test_validation_rejects_percent_typed_as_whole_numbers():
    problems = dcf.validate({**ASSUME, "risk_free_rate": 4.0, "tax_rate": 25})
    assert any("risk_free_rate" in p for p in problems) and any("tax_rate" in p for p in problems)


# --------------------------------------------------------------------------- reverse DCF
def test_implied_terminal_growth_round_trips_to_the_price():
    imp = dcf.implied_assumptions(BASE, ASSUME)
    g = imp["implied_terminal_growth"]
    assert g is not None
    v = dcf.value(BASE, {**ASSUME, "terminal_growth": g})["value_per_share"]
    assert v == pytest.approx(BASE["price"], rel=1e-5)


def test_implied_revenue_growth_round_trips_to_the_price():
    imp = dcf.implied_assumptions(BASE, ASSUME)
    r = imp["implied_revenue_growth"]
    assert r is not None
    v = dcf.value(BASE, {**ASSUME, "revenue_growth": [r] * 5})["value_per_share"]
    assert v == pytest.approx(BASE["price"], rel=1e-5)


def test_unreachable_price_gives_none_not_a_fake_answer():
    imp = dcf.implied_assumptions({**BASE, "price": 1e9}, ASSUME)
    assert imp["implied_terminal_growth"] is None


@pytest.mark.parametrize("override, fragment", [
    ({"revenue_growth": [-1.0, 0.0, 0.0, 0.0, 0.0]}, "revenue_growth values"),
    ({"revenue_growth": [5.0] * 5}, "revenue_growth values"),
    ({"ebit_margin": 1.5}, "ebit_margin"),
    ({"capex_pct_revenue": -0.1}, "capex_pct_revenue"),
    ({"terminal_growth": 2.5}, "terminal_growth"),
])
def test_pathological_assumptions_are_refused(override, fragment):
    problems = dcf.validate({**ASSUME, **override})
    assert any(fragment in p for p in problems)


@pytest.mark.parametrize("key", ["revenue", "shares", "price"])
def test_non_positive_base_values_are_refused(key):
    with pytest.raises(dcf.AssumptionError, match="not positive"):
        dcf.value({**BASE, key: 0.0}, ASSUME)


def test_assumption_sources_are_listed_and_gaps_reported():
    a = {**ASSUME, "sources": {"risk_free_rate": "10-year Treasury yield, 2025-12-31"}}
    assert "risk_free_rate" not in dcf.missing_sources(a) and "equity_risk_premium" in dcf.missing_sources(a)
    text = "\n".join(dcf.render_sources(a))
    assert "10-year Treasury yield, 2025-12-31" in text and "**not recorded**" in text


# --------------------------------------------------------------------------- identity and dates (standalone DCF)
def _filled(demo_run, tmp_path, ticker, as_of=None, name="a.yaml"):
    path = dcf.init_assumptions(demo_run, ticker, as_of, tmp_path / name)
    doc = yaml.safe_load(path.read_text())
    doc.update(ASSUME)
    doc.pop("scenarios", None)
    path.write_text(yaml.safe_dump(doc))
    return path, doc


def _nonfinancial(demo_run, k=0):
    s = pd.read_csv(f"{demo_run['project']['output_dir']}/screen_composite_latest.csv")
    return s[s["sector"] != "Financials"]["ticker"].iloc[k]


def test_standalone_dcf_refuses_a_different_valuation_month(demo_run, tmp_path):
    t = _nonfinancial(demo_run)
    path, doc = _filled(demo_run, tmp_path, t)                     # dated to the latest month
    with pytest.raises(dcf.AssumptionError, match=f"dated {doc['as_of']} but the valuation was requested as of 2016-06"):
        dcf.run(demo_run, t, path, as_of="2016-06")
    with pytest.raises(SystemExit, match="Cannot value"):
        dcf.main(["--ticker", t, "--mode", "demo", "--assumptions", str(path), "--as-of", "2016-06"])


def test_standalone_dcf_refuses_another_companys_assumptions(demo_run, tmp_path):
    t, other = _nonfinancial(demo_run, 0), _nonfinancial(demo_run, 1)
    path, _ = _filled(demo_run, tmp_path, t)
    with pytest.raises(dcf.AssumptionError, match=f"is for {t}, not {other}"):
        dcf.run(demo_run, other, path)


def test_matching_historical_assumptions_value_as_of_their_month(demo_run, tmp_path):
    t = _nonfinancial(demo_run)
    path, _ = _filled(demo_run, tmp_path, t, as_of="2016-06")
    out = dcf.run(demo_run, t, path, as_of="2016-06")
    assert out.name == f"{t}_201606.md" and "Valuation month-end: **2016-06-30**" in out.read_text()


def test_retrospective_valuation_is_labelled(demo_run, tmp_path):
    t = _nonfinancial(demo_run)
    path, doc = _filled(demo_run, tmp_path, t, as_of="2016-06")    # prepared_on = today, years later
    assert "Retrospective valuation" in dcf.run(demo_run, t, path).read_text()
    doc["prepared_on"] = "2016-07-20"                               # within the 45-day window
    path.write_text(yaml.safe_dump(doc))
    assert "Retrospective valuation" not in dcf.run(demo_run, t, path).read_text()


def test_retrospective_rule_boundaries():
    m = pd.Timestamp("2016-06-30")
    assert not dcf.retrospective("2016-08-14", m, 45) and dcf.retrospective("2016-08-15", m, 45)
    assert not dcf.retrospective(None, m, 45)


# --------------------------------------------------------------------------- scenario probabilities
@pytest.mark.parametrize("weights", [
    {"bear": -0.25, "base": 1.25},            # the audit's reproduction: sums to 1, but not probabilities
    {"bear": 1.5, "base": -0.5},
    {"bear": float("nan"), "base": 1.0},
    {"bear": "0.5", "base": 0.5},
    {"bear": True, "base": 0.0},
])
def test_scenario_weights_must_be_probabilities(weights):
    a = {**ASSUME, "scenarios": {"bear": {"ebit_margin": 0.10}}, "scenario_weights": weights}
    with pytest.raises(dcf.AssumptionError, match="probabilities between 0 and 1"):
        dcf.value_scenarios(BASE, a)


def test_valid_weights_keep_the_weighted_value_inside_the_scenario_range():
    a = {**ASSUME, "scenarios": {"bear": {"ebit_margin": 0.10}, "bull": {"ebit_margin": 0.20}},
         "scenario_weights": {"bear": 0.0, "base": 0.7, "bull": 0.3}}
    s = dcf.value_scenarios(BASE, a)
    assert s["value_per_share"].min() <= dcf.weighted_value(s) <= s["value_per_share"].max()


# --------------------------------------------------------------------------- mandatory identity metadata
@pytest.mark.parametrize("drop", ["ticker", "as_of"])
def test_files_without_identity_metadata_are_refused(demo_run, tmp_path, drop):
    t = _nonfinancial(demo_run)
    path, doc = _filled(demo_run, tmp_path, t)
    doc.pop(drop)
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(dcf.AssumptionError, match=f"has no {drop}"):
        dcf.run(demo_run, t, path)
    with pytest.raises(dcf.AssumptionError, match=f"has no {drop}"):
        dcf.run(demo_run, t, path, as_of="2016-06")          # an explicit date does not stand in for it


def test_malformed_dates_are_refused(demo_run, tmp_path):
    t = _nonfinancial(demo_run)
    path, doc = _filled(demo_run, tmp_path, t)
    doc["prepared_on"] = "sometime in 2026"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(dcf.AssumptionError, match="must be dates"):
        dcf.run(demo_run, t, path)


def test_missing_prepared_on_is_labelled_undated(demo_run, tmp_path):
    t = _nonfinancial(demo_run)
    path, doc = _filled(demo_run, tmp_path, t)
    doc.pop("prepared_on")
    path.write_text(yaml.safe_dump(doc))
    text = dcf.run(demo_run, t, path).read_text()
    assert "**Undated assumptions.**" in text and "Retrospective valuation" not in text


# --------------------------------------------------------------------------- non-finite inputs and dates
NAN = float("nan")


@pytest.mark.parametrize("override", [
    {"revenue_growth": [NAN, 0.03, 0.03, 0.03, 0.03]},
    {"ebit_margin": NAN},
    {"tax_rate": float("inf")},
    {"terminal_growth": NAN},
    {"risk_free_rate": "0.04"},
    {"capex_pct_revenue": True},
    {"beta": NAN},
    {"total_debt": NAN},
    {"cash": -5.0},
])
def test_non_finite_or_non_numeric_assumptions_are_refused(override):
    with pytest.raises(dcf.AssumptionError):
        dcf.value(BASE, {**ASSUME, **override})


def test_invalid_scenario_is_refused_not_dropped_from_the_weighted_value():
    a = {**ASSUME, "scenarios": {"bear": {"revenue_growth": [NAN] * 5}}, "scenario_weights": {"bear": 0.5, "base": 0.5}}
    with pytest.raises(dcf.AssumptionError, match="scenario 'bear'"):
        dcf.value_scenarios(BASE, a)


def test_weighted_value_refuses_any_invalid_scenario_value():
    scen = pd.DataFrame({"scenario": ["bear", "base"], "value_per_share": [NAN, 21.83], "weight": [0.5, 0.5]})
    with pytest.raises(dcf.AssumptionError, match="weighted value"):
        dcf.weighted_value(scen)


def test_non_finite_base_beta_without_override_is_refused():
    with pytest.raises(dcf.AssumptionError, match="beta"):
        dcf.value({**BASE, "beta_60m": NAN}, ASSUME)


@pytest.mark.parametrize("meta", [
    {"ticker": "T", "as_of": "NaT"},
    {"ticker": "T", "as_of": "nan"},
    {"ticker": "T", "as_of": "2025-12", "prepared_on": "NaT"},
    {"ticker": "T", "as_of": "2025-12", "prepared_on": "not a date"},
])
def test_nat_and_invalid_dates_are_refused(meta):
    with pytest.raises(dcf.AssumptionError, match="must be dates"):
        dcf.check_identity(meta, "T")


def test_valid_and_absent_dates():
    assert dcf.check_identity({"ticker": "T", "as_of": "2025-12"}, "T") == pd.Timestamp("2025-12-31")
    assert dcf.check_identity({"ticker": "T", "as_of": "2025-12", "prepared_on": "2026-01-10"}, "T") == \
        pd.Timestamp("2025-12-31")
    assert dcf.parse_date(None) is None and dcf.parse_date("") is None


# --------------------------------------------------------------------------- sensitivity-grid settings
@pytest.mark.parametrize("setting", [
    {"wacc_step": NAN}, {"growth_step": float("inf")}, {"wacc_step": 0.0}, {"growth_step": -0.005},
    {"wacc_step": 0.5}, {"steps": -2}, {"steps": "oops"}, {"steps": 2.5}, {"steps": True}, {"steps": 50},
])
def test_invalid_sensitivity_settings_are_refused(setting):
    a = {**ASSUME, "sensitivity": {"wacc_step": 0.005, "growth_step": 0.005, "steps": 2, **setting}}
    with pytest.raises(dcf.AssumptionError, match="sensitivity"):
        dcf.value(BASE, a)                                   # refused up front, with the other assumptions
    with pytest.raises(dcf.AssumptionError, match="sensitivity"):
        dcf.sensitivity(BASE, a, 0.09)                       # and if the grid is built directly


def test_valid_sensitivity_settings_shape_the_grid():
    w = dcf.value(BASE, ASSUME)["wacc"]
    assert dcf.sensitivity(BASE, {**ASSUME, "sensitivity": {"steps": 0}}, w).shape == (1, 1)
    assert dcf.sensitivity(BASE, {**ASSUME, "sensitivity": {"steps": 3, "wacc_step": 0.0025}}, w).shape == (7, 7)
    assert dcf.sensitivity(BASE, ASSUME, w).shape == (5, 5)  # block absent: defaults
    assert dcf.validate_sensitivity({"sensitivity": "wide"}) != []
