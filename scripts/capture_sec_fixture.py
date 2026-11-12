"""Save a trimmed SEC companyfacts file as a real-data test fixture.

Run once on a machine with internet access, then commit the JSON it writes:

    python scripts/capture_sec_fixture.py --ticker AAPL --user-agent "Your Name you@example.com"
    python scripts/capture_sec_fixture.py --ticker NVDA --user-agent "Your Name you@example.com"
    python -m pytest tests/test_real_fixture.py -v

Only the us-gaap concepts the pipeline reads are kept, so the file stays small.
SEC makes EDGAR data freely available; check https://www.sec.gov/privacy for its
current terms before publishing the fixture. Prices (Yahoo) are NOT saved, because
Yahoo's terms do not allow redistribution.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from eqscreen.fundamentals import SEC_CONCEPTS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--user-agent", required=True, help="SEC requires your name and email")
    ap.add_argument("--cik", help="10-digit CIK; looked up from SEC if omitted")
    args = ap.parse_args()
    headers = {"User-Agent": args.user_agent}
    ticker = args.ticker.upper()

    cik = args.cik
    if not cik:
        mapping = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30).json()
        cik = next(str(r["cik_str"]).zfill(10) for r in mapping.values()
                   if str(r["ticker"]).upper().replace(".", "-") == ticker)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    facts = resp.json()

    wanted = {concept for candidates in SEC_CONCEPTS.values() for concept, _ in candidates}
    gaap = facts.get("facts", {}).get("us-gaap", {})
    trimmed = {
        "_fixture": {"ticker": ticker, "cik": cik, "source_url": url,
                     "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "entityName": facts.get("entityName"),
        "facts": {"us-gaap": {k: v for k, v in gaap.items() if k in wanted}},
    }
    out = ROOT / "tests" / "fixtures" / f"sec_companyfacts_{ticker}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trimmed), encoding="utf-8")
    print(f"Saved {len(trimmed['facts']['us-gaap'])} concepts to {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
