"""Real-data checks on saved SEC filings. Skipped until a fixture is captured:

    python scripts/capture_sec_fixture.py --ticker AAPL --user-agent "Your Name you@example.com"

What is checked on real filings (no numbers are hard-coded; only the publicly
documented split dates and ratios are):
  * at least one fiscal year has an original filing and a later revised version,
    each tied to a well-formed accession number and EDGAR link;
  * the pre-split share count, once split-adjusted, lines up with the first
    post-split filing (continuity across the split);
  * the issuer's own restated comparative for that pre-split year shows the share
    count multiplied by the split ratio, which cross-checks the split factor.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from eqscreen.factors import split_factor
from eqscreen.fundamentals import edgar_filing_url, parse_sec_companyfacts

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("sec_companyfacts_*.json"))
# Effective dates of real splits as announced by the companies.
KNOWN_SPLITS = {
    "AAPL": [("2020-08-31", 4.0)],
    "NVDA": [("2021-07-20", 4.0), ("2024-06-10", 10.0)],
    "TSLA": [("2020-08-31", 5.0), ("2022-08-25", 3.0)],
}

if not FIXTURES:
    pytest.skip("no real SEC fixture captured yet (see scripts/capture_sec_fixture.py)", allow_module_level=True)


def _load(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = raw["_fixture"]
    return meta, parse_sec_companyfacts(raw, meta["ticker"], 90)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_original_and_revised_versions_with_accession_links(path):
    meta, df = _load(path)
    assert len(df) >= 5, "expected several years of annual filings"
    assert df["accession"].str.fullmatch(r"\d{10}-\d{2}-\d{6}").all()
    assert (df["available_date"] > df["period_end"]).all()
    versions = df.groupby("period_end").size()
    revised = versions[versions > 1].index
    assert len(revised), "no fiscal year was ever revised; pick a company with a known restatement or split"
    first = df[df["period_end"] == revised[0]].sort_values("available_date")
    assert first.iloc[0]["version_type"] == "original"
    assert set(first.iloc[1:]["version_type"]) <= {"amendment", "restatement"}
    url = edgar_filing_url(meta["cik"], first.iloc[0]["accession"])
    assert re.fullmatch(r"https://www\.sec\.gov/Archives/edgar/data/\d+/\d{18}/", url)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_real_split_is_continuous_after_adjustment(path):
    meta, df = _load(path)
    known = KNOWN_SPLITS.get(meta["ticker"])
    if not known:
        pytest.skip(f"no documented split listed for {meta['ticker']}")
    splits = pd.DataFrame({"ticker": meta["ticker"], "split_date": pd.to_datetime([d for d, _ in known]),
                           "ratio": [r for _, r in known]})
    df = df.dropna(subset=["shares_diluted"]).copy()
    df["shares_adj"] = df["shares_diluted"] * split_factor(df, splits, "available_date")
    originals = df[df["version_type"] == "original"].sort_values("available_date")
    checked = 0
    for date, ratio in known:
        before = originals[originals["available_date"] < date]
        after = originals[originals["available_date"] >= date]
        if before.empty or after.empty:
            continue
        b, a = before.iloc[-1], after.iloc[0]
        jump = a["shares_adj"] / b["shares_adj"]
        assert 0.8 <= jump <= 1.25, f"{meta['ticker']} {date}: adjusted shares change {jump:.2f}x across the split"
        # The issuer restated that pre-split year in its next 10-K: shares x ratio.
        later = df[(df["period_end"] == b["period_end"]) & (df["available_date"] > date)]
        if len(later):
            assert later.iloc[0]["shares_diluted"] / b["shares_diluted"] == pytest.approx(ratio, rel=0.1)
        checked += 1
    if not checked:
        pytest.skip("fixture history does not span a documented split")
