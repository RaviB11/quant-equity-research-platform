import pandas as pd
import pytest
import yaml

from eqscreen import dcf
from eqscreen.research import RISKS_TEMPLATE, build_report, risks_written

ASSUME = {"revenue_growth": [0.04] * 5, "ebit_margin": 0.2, "tax_rate": 0.21, "da_pct_revenue": 0.05,
          "capex_pct_revenue": 0.06, "nwc_pct_incremental_revenue": 0.1, "terminal_growth": 0.02,
          "risk_free_rate": 0.04, "equity_risk_premium": 0.05, "pre_tax_cost_of_debt": 0.055,
          "scenarios": {"bear": {"revenue_growth": [0.0] * 5, "ebit_margin": 0.15},
                        "bull": {"revenue_growth": [0.07] * 5, "ebit_margin": 0.24}}}
SECTIONS = ("Summary", "Financial fundamentals", "Peer valuation", "DCF valuation", "Earnings quality",
            "Factor rankings", "Price risk", "Key risks", "Sources", "Model limitations")


def _ticker(demo_run, demo_conn):
    screen = pd.read_csv(f"{demo_run['project']['output_dir']}/screen_composite_latest.csv")
    joiners = set(pd.read_sql_query("SELECT ticker FROM membership WHERE start_date IS NOT NULL", demo_conn)["ticker"])
    return screen[(screen["sector"] != "Financials") & ~screen["ticker"].isin(joiners)]["ticker"].iloc[0]


SIGNED = "Reviewed by: R. Bisht, 2026-11-12\n"


def test_risks_need_text_and_a_dated_sign_off():
    assert not risks_written(RISKS_TEMPLATE.format(ticker="X"))
    text = "### Business\n- Revenue depends on two customers, which together were 40% of sales (10-K 2025)."
    assert not risks_written(text)                                  # no sign-off: still a draft
    assert not risks_written(text + "\nReviewed by: \n")             # empty sign-off
    assert not risks_written(text + "\nReviewed by: R. Bisht\n")      # no date
    assert risks_written(text + "\n" + SIGNED)
    assert not risks_written("### Business\n- ok\n" + SIGNED)        # sign-off alone is not enough


def test_report_is_draft_without_dcf_or_risks(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    text, meta = build_report(demo_run, t, assumptions_path=tmp_path / "none.yaml", risks_path=tmp_path / "none.md")
    assert meta["draft"] and not meta["dcf"]
    assert "**DRAFT.**" in text and "DCF not run" in text
    for s in SECTIONS:
        assert f"## {s}" in text


def test_full_report_with_scenarios_and_risks(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    a_path = dcf.init_assumptions(demo_run, t, None, tmp_path / f"{t}.yaml")
    doc = yaml.safe_load(a_path.read_text())
    doc.update(ASSUME)
    a_path.write_text(yaml.safe_dump(doc))
    r_path = tmp_path / "risks.md"
    r_path.write_text("### Business\n- Placeholder risk text long enough to count as written for this template test.\n"
                      + SIGNED)
    text, meta = build_report(demo_run, t, assumptions_path=a_path, risks_path=r_path)
    assert not meta["draft"] and meta["dcf"]
    assert text.startswith("> **SYNTHETIC DEMO DATA.**") and "**DRAFT.**" not in text
    assert "### Scenarios" in text and "| Bear |" in text and "| Bull |" in text
    assert "### What the share price implies (reverse DCF)" in text
    assert "#### Business" in text                                   # analyst headings nested
    assert "## DCF assumptions (base case)" in text
    assert f"| **{t}** |" in text and "Sector median" in text


def test_report_respects_as_of(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    text, meta = build_report(demo_run, t, as_of="2016-06", assumptions_path=tmp_path / "x.yaml",
                              risks_path=tmp_path / "x.md")
    assert meta["as_of"] == pd.Timestamp("2016-06-30")
    fy_header = [ln for ln in text.splitlines() if ln.startswith("| Fiscal year |")][0]
    assert "2015-12" in fy_header and "2016-12" not in fy_header


def test_html_rendering_is_self_contained(tmp_path):
    from eqscreen.render import write_html

    md = tmp_path / "r.md"
    md.write_text("> **SYNTHETIC DEMO DATA.**\n\n# Title\n\n| A | B |\n|---|---:|\n| x | 1.0 |\n\n- item [link](https://www.sec.gov/)\n")
    html = write_html(md, "T", "footer text").read_text()
    assert "<table>" in html and "<blockquote>" in html and "footer text" in html
    assert 'text-align: right' in html                                   # numeric column alignment kept
    assert "<script" not in html and "fonts.googleapis" not in html and "<link" not in html


def test_historical_report_never_includes_a_later_dcf(demo_run, demo_conn, tmp_path):
    """Audit P1-A: a 2016 report must not embed a DCF valued at 2025-12."""
    t = _ticker(demo_run, demo_conn)
    a_path = dcf.init_assumptions(demo_run, t, None, tmp_path / f"{t}.yaml")      # dated latest month
    doc = yaml.safe_load(a_path.read_text())
    doc.update(ASSUME)
    a_path.write_text(yaml.safe_dump(doc))
    text, meta = build_report(demo_run, t, as_of="2016-06", assumptions_path=a_path, risks_path=tmp_path / "x.md")
    assert meta["as_of"] == pd.Timestamp("2016-06-30") and not meta["dcf"]
    assert "DCF omitted: the assumptions file is dated" in text
    assert "### Scenarios" not in text and "Value per share" not in text
    # The only later date allowed is the note naming the file's month; no later data or valuation.
    body = text.replace(f"dated {doc['as_of']}", "").replace(f"--as-of {doc['as_of']}", "")
    assert doc["as_of"][:4] + "-" not in body.split("Run:")[0] + body.split("## Summary")[1].split("## Sources")[0]

    # Same file, matching month: the DCF is included.
    text2, meta2 = build_report(demo_run, t, as_of=doc["as_of"], assumptions_path=a_path, risks_path=tmp_path / "x.md")
    assert meta2["dcf"] and "### Scenarios" in text2

    # Assumptions dated to the historical month value as of that month.
    old = dcf.init_assumptions(demo_run, t, "2016-06", tmp_path / "old.yaml")
    d_old = yaml.safe_load(old.read_text())
    d_old.update(ASSUME)
    old.write_text(yaml.safe_dump(d_old))
    text3, meta3 = build_report(demo_run, t, as_of="2016-06", assumptions_path=old, risks_path=tmp_path / "x.md")
    assert meta3["dcf"] and "Valuation month 2016-06-30" in text3



def test_uncited_risks_and_unsourced_dcf_inputs_are_flagged(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    a_path = dcf.init_assumptions(demo_run, t, None, tmp_path / f"{t}.yaml")
    doc = yaml.safe_load(a_path.read_text())
    doc.update(ASSUME)
    doc["sources"]["risk_free_rate"] = "US 10-year Treasury, Federal Reserve H.15, as of the valuation date"
    a_path.write_text(yaml.safe_dump(doc))
    r_path = tmp_path / "risks.md"
    r_path.write_text("### Business\n- Customer concentration and pricing pressure in a simulated market for this test.\n"
                      + SIGNED)
    text, meta = build_report(demo_run, t, assumptions_path=a_path, risks_path=r_path)
    assert not meta["draft"]
    assert "cites no sources" in text
    assert "DCF input(s) have no recorded source" in text and "risk_free_rate" not in text.split(
        "DCF input(s) have no recorded source:")[1].split("\n")[0]
    assert "### Sources of the assumptions" in text and "Federal Reserve H.15" in text


# --------------------------------------------------------------------------- hindsight (retrospective) labels
RISK_TEXT = "### Business\n- Customer concentration and pricing pressure, per the 10-K risk factors, for this test.\n"


def test_historical_report_labels_later_risk_review_as_retrospective(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    r = tmp_path / "risks.md"
    r.write_text(RISK_TEXT + "Reviewed by: R. Bisht, 2026-11-12\n")
    text, meta = build_report(demo_run, t, as_of="2016-06", assumptions_path=tmp_path / "x.yaml", risks_path=r)
    assert not meta["draft"] and meta["retrospective_risks"]
    assert "**Retrospective analysis.**" in text and "key risks reviewed on 2026-11-12" in text
    assert "### Analyst assessment (retrospective: reviewed 2026-11-12, after the report date)" in text


def test_current_report_signed_shortly_after_month_end_is_not_retrospective(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    r = tmp_path / "risks.md"
    r.write_text(RISK_TEXT + "Reviewed by: R. Bisht, 2026-01-20\n")        # 20 days after 2025-12-31
    text, meta = build_report(demo_run, t, assumptions_path=tmp_path / "x.yaml", risks_path=r)
    assert not meta["retrospective_risks"] and "Retrospective" not in text
    r.write_text(RISK_TEXT + "Reviewed by: R. Bisht, 2026-11-12\n")        # months later
    assert build_report(demo_run, t, assumptions_path=tmp_path / "x.yaml", risks_path=r)[1]["retrospective_risks"]


def test_research_omits_dcf_written_for_another_company(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    a_path = dcf.init_assumptions(demo_run, t, None, tmp_path / "a.yaml")
    doc = yaml.safe_load(a_path.read_text())
    doc.update(ASSUME)
    doc["ticker"] = "OTHER"
    a_path.write_text(yaml.safe_dump(doc))
    text, meta = build_report(demo_run, t, assumptions_path=a_path, risks_path=tmp_path / "x.md")
    assert not meta["dcf"] and f"DCF omitted: the assumptions file is for OTHER, not {t}." in text


def test_research_handles_incomplete_dcf_metadata(demo_run, demo_conn, tmp_path):
    t = _ticker(demo_run, demo_conn)
    a_path = dcf.init_assumptions(demo_run, t, None, tmp_path / "a.yaml")
    doc = yaml.safe_load(a_path.read_text())
    doc.update(ASSUME)
    doc.pop("as_of")
    a_path.write_text(yaml.safe_dump(doc))
    text, meta = build_report(demo_run, t, assumptions_path=a_path, risks_path=tmp_path / "x.md")
    assert not meta["dcf"] and "DCF omitted: the assumptions file has no as_of" in text

    doc["as_of"] = str(meta["as_of"].strftime("%Y-%m"))
    doc.pop("prepared_on")
    a_path.write_text(yaml.safe_dump(doc))
    text, meta = build_report(demo_run, t, assumptions_path=a_path, risks_path=tmp_path / "x.md")
    assert meta["dcf"] and meta["undated_dcf"]
    assert "**Undated DCF assumptions.**" in text and "DCF assumptions are undated" in text
