"""Provider parsers, tested offline against payloads shaped like each API's documented format.

All numbers here are invented fixture values, not real company data.
"""
import numpy as np
import pandas as pd
import pytest

from eqscreen.fundamentals import (FUND_COLS, RateLimitError, parse_alphavantage, parse_fmp,
                                   parse_sec_companyfacts, parse_yfinance)
from eqscreen.universe import parse_wikipedia_table, to_yahoo_symbol


# --------------------------------------------------------------------------- FMP
def test_fmp_stable_fields():
    income = [
        {"date": "2023-12-31", "filingDate": "2024-02-15", "revenue": 1000, "ebit": 200, "operatingIncome": 190,
         "ebitda": 260, "netIncome": 150, "epsDiluted": 1.5, "weightedAverageShsOutDil": 100},
        {"date": "2022-12-31", "filingDate": "2023-02-14", "revenue": 900, "ebit": 180, "operatingIncome": 170,
         "ebitda": 230, "netIncome": 120, "epsDiluted": 1.2, "weightedAverageShsOutDil": 100},
    ]
    balance = [
        {"date": "2023-12-31", "totalStockholdersEquity": 800, "totalDebt": 300, "cashAndCashEquivalents": 50},
        {"date": "2022-12-31", "totalStockholdersEquity": 700, "totalDebt": 320, "cashAndCashEquivalents": 40},
    ]
    df = parse_fmp(income, balance, "TEST", 90)
    assert list(df.columns) == FUND_COLS
    assert list(df["period_end"]) == ["2022-12-31", "2023-12-31"]
    last = df.iloc[-1]
    assert last["available_date"] == "2024-02-15" and last["available_date_is_estimated"] == 0
    assert last["ebit"] == 200 and last["eps_diluted"] == 1.5 and last["total_debt"] == 300


def test_fmp_legacy_field_names():
    income = [{"date": "2023-12-31", "fillingDate": "2024-02-20", "revenue": 1000, "operatingIncome": 190,
               "ebitda": 250, "netIncome": 150, "epsdiluted": 1.5, "weightedAverageShsOutDil": 100}]
    df = parse_fmp(income, [], "TEST", 90)
    assert df.iloc[0]["available_date"] == "2024-02-20"
    assert df.iloc[0]["ebit"] == 190          # falls back to operatingIncome
    assert df.iloc[0]["eps_diluted"] == 1.5
    assert np.isnan(df.iloc[0]["total_equity"])


def test_filing_date_before_period_end_falls_back_to_lag():
    income = [{"date": "2023-12-31", "filingDate": "2023-01-01", "revenue": 1, "netIncome": 1}]
    df = parse_fmp(income, [], "TEST", 90)
    assert df.iloc[0]["available_date"] == "2024-03-30"
    assert df.iloc[0]["available_date_is_estimated"] == 1


# --------------------------------------------------------------------------- Alpha Vantage
def test_alphavantage_strings_none_and_debt_fallback():
    income = {"symbol": "TEST", "annualReports": [
        {"fiscalDateEnding": "2023-12-31", "totalRevenue": "1000", "ebit": "200", "ebitda": "None", "netIncome": "150"}]}
    balance = {"symbol": "TEST", "annualReports": [
        {"fiscalDateEnding": "2023-12-31", "totalShareholderEquity": "800", "shortLongTermDebtTotal": "None",
         "longTermDebt": "250", "shortTermDebt": "50", "cashAndCashEquivalentsAtCarryingValue": "50",
         "commonStockSharesOutstanding": "100"}]}
    row = parse_alphavantage(income, balance, "TEST", 90).iloc[0]
    assert row["total_debt"] == 300
    assert np.isnan(row["ebitda"]) and np.isnan(row["eps_diluted"])
    assert row["shares_diluted"] == 100
    assert row["available_date"] == "2024-03-30" and row["available_date_is_estimated"] == 1


@pytest.mark.parametrize("payload", [{"Information": "rate limit"}, {"Note": "Thank you for using Alpha Vantage"}])
def test_alphavantage_rate_limit(payload):
    with pytest.raises(RateLimitError):
        parse_alphavantage(payload, payload, "TEST", 90)


# --------------------------------------------------------------------------- SEC EDGAR
def _fact(start, end, val, filed, form="10-K"):
    e = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(filed[:4]), "fp": "FY"}
    if start:
        e["start"] = start
    return e


SEC_FACTS = {"facts": {"us-gaap": {
    "NetIncomeLoss": {"units": {"USD": [
        _fact("2022-01-01", "2022-12-31", 100, "2023-02-10"),
        _fact("2022-01-01", "2022-12-31", 110, "2024-02-09"),            # restated comparative: must be ignored
        _fact("2023-01-01", "2023-12-31", 120, "2024-02-09"),
        _fact("2023-07-01", "2023-09-30", 30, "2023-11-01", form="10-Q"),  # quarterly 10-Q: ignored
        _fact("2023-10-01", "2023-12-31", 35, "2024-02-09"),             # quarter inside a 10-K: ignored
    ]}},
    "Revenues": {"units": {"USD": [_fact("2022-01-01", "2022-12-31", 1000, "2023-02-10"),
                                   _fact("2023-01-01", "2023-12-31", 1100, "2024-02-09")]}},
    "OperatingIncomeLoss": {"units": {"USD": [_fact("2023-01-01", "2023-12-31", 200, "2024-02-09")]}},
    "DepreciationDepletionAndAmortization": {"units": {"USD": [_fact("2023-01-01", "2023-12-31", 40, "2024-02-09")]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [_fact("2023-01-01", "2023-12-31", 1.2, "2024-02-09")]}},
    "StockholdersEquity": {"units": {"USD": [_fact(None, "2023-12-31", 800, "2024-02-09"),
                                             _fact(None, "2023-06-30", 750, "2024-02-09")]}},
    "LongTermDebtNoncurrent": {"units": {"USD": [_fact(None, "2023-12-31", 250, "2024-02-09")]}},
    "LongTermDebtCurrent": {"units": {"USD": [_fact(None, "2023-12-31", 50, "2024-02-09")]}},
    "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_fact(None, "2023-12-31", 60, "2024-02-09")]}},
}}}


def test_sec_versions_original_then_restatement():
    df = parse_sec_companyfacts(SEC_FACTS, "TEST", 90)
    assert set(df["period_end"]) == {"2022-12-31", "2023-12-31"}   # no 2023-06-30 or quarter rows
    fy22 = df[df["period_end"] == "2022-12-31"].sort_values("available_date")
    assert fy22["net_income"].tolist() == [100, 110]                 # original, then restated comparative
    assert fy22["available_date"].tolist() == ["2023-02-10", "2024-02-09"]
    assert fy22["version_type"].tolist() == ["original", "restatement"]
    fy23 = df[df["period_end"] == "2023-12-31"].iloc[0]
    assert len(df[df["period_end"] == "2023-12-31"]) == 1
    assert fy23["net_income"] == 120 and fy23["available_date"] == "2024-02-09"
    assert fy23["ebitda"] == 240 and fy23["total_debt"] == 300 and fy23["total_equity"] == 800
    assert fy23["available_date_is_estimated"] == 0


def test_sec_amendment_is_a_new_dated_version():
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            {**_fact("2023-01-01", "2023-12-31", 120, "2024-02-09"), "accn": "A-1"},
            {**_fact("2023-01-01", "2023-12-31", 60, "2024-05-20", form="10-K/A"), "accn": "A-2"}]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {**_fact("2023-01-01", "2023-12-31", 1.2, "2024-02-09"), "accn": "A-1"},
            {**_fact("2023-01-01", "2023-12-31", 0.6, "2024-05-20", form="10-K/A"), "accn": "A-2"}]}},
    }}}
    df = parse_sec_companyfacts(facts, "TEST", 90).sort_values("available_date")
    assert df["eps_diluted"].tolist() == [1.2, 0.6]
    assert df["version_type"].tolist() == ["original", "amendment"]
    assert df["accession"].tolist() == ["A-1", "A-2"]


def test_sec_identical_comparative_does_not_create_a_version():
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        _fact("2022-01-01", "2022-12-31", 100, "2023-02-10"),
        _fact("2022-01-01", "2022-12-31", 100, "2024-02-09")]}}}}}
    assert len(parse_sec_companyfacts(facts, "TEST", 90)) == 1


# --------------------------------------------------------------------------- yfinance
def test_yfinance_raw_keys_and_ebitda_fallback():
    cols = pd.to_datetime(["2023-12-31", "2022-12-31"])
    inc = pd.DataFrame({cols[0]: [1000, 200, 150, 1.5, 100], cols[1]: [900, 180, 120, 1.2, 100]},
                       index=["TotalRevenue", "EBIT", "NetIncome", "DilutedEPS", "DilutedAverageShares"])
    bs = pd.DataFrame({cols[0]: [800, 300, 50], cols[1]: [700, 320, 40]},
                      index=["StockholdersEquity", "TotalDebt", "CashAndCashEquivalents"])
    cf = pd.DataFrame({cols[0]: [40], cols[1]: [35]}, index=["DepreciationAndAmortization"])
    df = parse_yfinance(inc, bs, cf, "TEST", 90).set_index("period_end")
    assert df.loc["2023-12-31", "ebitda"] == 240
    assert df.loc["2022-12-31", "total_equity"] == 700
    assert (df["available_date_is_estimated"] == 1).all()


# --------------------------------------------------------------------------- universe
def test_wikipedia_table_parsing():
    html = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
    <th>GICS Sub-Industry</th><th>CIK</th></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Multi-Sector Holdings</td><td>1067983</td></tr>
    <tr><td>TST</td><td>Test Co</td><td>Industrials</td><td>Machinery</td><td>12</td></tr></table>"""
    df = parse_wikipedia_table(html)
    assert list(df["ticker"]) == ["BRK-B", "TST"]
    assert df.loc[0, "cik"] == "0001067983"
    assert to_yahoo_symbol(" brk.b ") == "BRK-B"


# --------------------------------------------------------------------------- request logging
class _Resp:
    def __init__(self, code, payload=None):
        self.status_code, self._payload = code, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_every_http_attempt_is_logged_without_api_key(monkeypatch):
    from eqscreen import fundamentals as f

    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    logged = []
    session = _Session([_Resp(429), _Resp(503), _Resp(200, {"ok": 1})])
    out = f._get_json(session, "https://api.example.com/stable/income-statement", {"apikey": "SECRET"},
                      on_request=lambda *a: logged.append(a))
    assert out == {"ok": 1}
    assert [s for _, s, _ in logged] == ["request_retry", "request_retry", "request_ok"]
    assert all("SECRET" not in e and "SECRET" not in m for e, _, m in logged)
    assert logged[1][2] == "HTTP 503, attempt 2/3"


def test_rate_limit_after_retries_is_logged_and_raised(monkeypatch):
    from eqscreen import fundamentals as f

    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    logged = []
    with pytest.raises(f.RateLimitError):
        f._get_json(_Session([_Resp(429)] * 3), "https://x/y", {}, on_request=lambda *a: logged.append(a))
    assert [s for _, s, _ in logged] == ["request_retry", "request_retry", "request_failed"]


def test_fetch_loop_logs_requests_outcomes_and_reuses_cache(fresh_demo_db, monkeypatch):
    from eqscreen import fundamentals as f
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    cfg["fundamentals"]["source"] = "fmp"
    monkeypatch.setenv("FMP_API_KEY", "SECRET")
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    income = [{"date": "2023-12-31", "filingDate": "2024-02-15", "revenue": 10, "netIncome": 1,
               "epsDiluted": 0.1, "weightedAverageShsOutDil": 10}]
    balance = [{"date": "2023-12-31", "totalStockholdersEquity": 5, "totalDebt": 2, "cashAndCashEquivalents": 1}]
    cashflow = [{"date": "2023-12-31", "operatingCashFlow": 1.5, "capitalExpenditure": -0.4}]
    session = _Session([_Resp(200, income), _Resp(429), _Resp(200, balance), _Resp(200, cashflow)])
    monkeypatch.setattr(f.requests, "Session", lambda: session)

    res = f.fetch_fundamentals(cfg, ["SYN001"], conn, "run-1", get_logger())
    assert res["ok"] == 1 and res["calls"] == 4          # every HTTP attempt, including the retry
    log = pd.read_sql_query("SELECT endpoint, ticker, status, message FROM fetch_log WHERE run_id = 'run-1'", conn)
    assert log["status"].tolist() == ["request_ok", "request_retry", "request_ok", "request_ok", "ok"]
    assert (log["ticker"] == "SYN001").all()
    assert not log["endpoint"].str.contains("SECRET").any() and not log["message"].str.contains("SECRET").any()
    stored = pd.read_sql_query("SELECT * FROM fundamentals WHERE source = 'fmp'", conn)
    assert len(stored) == 1 and stored.iloc[0]["available_date"] == "2024-02-15"
    assert stored.iloc[0]["operating_cash_flow"] == 1.5 and stored.iloc[0]["capex"] == 0.4

    # Second run within cache_days makes no HTTP calls.
    res2 = f.fetch_fundamentals(cfg, ["SYN001"], conn, "run-2", get_logger())
    assert res2["cached"] == 1 and res2["calls"] == 0 and len(session.calls) == 4



# --------------------------------------------------------------------------- cash-flow items
def test_cash_flow_items_and_capex_sign_per_provider():
    fmp = parse_fmp([{"date": "2023-12-31", "revenue": 1, "netIncome": 1}],
                    [{"date": "2023-12-31", "totalAssets": 50}], "T", 90,
                    [{"date": "2023-12-31", "operatingCashFlow": 12, "capitalExpenditure": -4}]).iloc[0]
    assert (fmp["operating_cash_flow"], fmp["capex"], fmp["total_assets"]) == (12, 4, 50)

    inc = {"annualReports": [{"fiscalDateEnding": "2023-12-31", "totalRevenue": "1", "netIncome": "1"}]}
    bal = {"annualReports": [{"fiscalDateEnding": "2023-12-31", "totalAssets": "50"}]}
    cf = {"annualReports": [{"fiscalDateEnding": "2023-12-31", "operatingCashflow": "12", "capitalExpenditures": "4"}]}
    av = parse_alphavantage(inc, bal, "T", 90, cf).iloc[0]
    assert (av["operating_cash_flow"], av["capex"], av["total_assets"]) == (12, 4, 50)

    col = pd.to_datetime(["2023-12-31"])
    yf_inc = pd.DataFrame({col[0]: [1, 1]}, index=["TotalRevenue", "NetIncome"])
    yf_bs = pd.DataFrame({col[0]: [50]}, index=["TotalAssets"])
    yf_cf = pd.DataFrame({col[0]: [12, -4]}, index=["OperatingCashFlow", "CapitalExpenditure"])
    y = parse_yfinance(yf_inc, yf_bs, yf_cf, "T", 90).iloc[0]
    assert (y["operating_cash_flow"], y["capex"], y["total_assets"]) == (12, 4, 50)

    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [_fact("2023-01-01", "2023-12-31", 1, "2024-02-09")]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact("2023-01-01", "2023-12-31", 12, "2024-02-09")]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fact("2023-01-01", "2023-12-31", 4, "2024-02-09")]}},
        "Assets": {"units": {"USD": [_fact(None, "2023-12-31", 50, "2024-02-09")]}},
    }}}
    sec = parse_sec_companyfacts(facts, "T", 90).iloc[0]
    assert (sec["operating_cash_flow"], sec["capex"], sec["total_assets"]) == (12, 4, 50)



@pytest.mark.parametrize("cap, http_calls", [(1, 1), (2, 2), (3, 3)])
def test_request_budget_counts_every_http_attempt(fresh_demo_db, monkeypatch, cap, http_calls):
    """Audit P2-D: a cap of N allows exactly N HTTP attempts, retries included."""
    from eqscreen import fundamentals as f
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    cfg["fundamentals"].update(source="fmp", max_requests_per_run=cap)
    monkeypatch.setenv("FMP_API_KEY", "SECRET")
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    ok = [{"date": "2023-12-31", "revenue": 1, "netIncome": 1}]
    session = _Session([_Resp(200, ok), _Resp(429), _Resp(200, ok), _Resp(200, ok), _Resp(200, ok)])
    monkeypatch.setattr(f.requests, "Session", lambda: session)
    res = f.fetch_fundamentals(cfg, ["SYN001", "SYN002"], conn, f"cap-{cap}", get_logger())
    assert len(session.calls) == http_calls and res["calls"] == http_calls
    assert res["ok"] == 0 and res["skipped"] == 2              # 4 attempts needed for the first ticker
    log = pd.read_sql_query(f"SELECT ticker, status FROM fetch_log WHERE run_id = 'cap-{cap}'", conn)
    assert (log["status"] == "budget_exhausted").sum() == 1 and (log["status"] == "skipped").sum() == 1
    assert conn.execute("SELECT COUNT(*) FROM fundamentals WHERE source = 'fmp'").fetchone()[0] == 0


def test_budget_object():
    from eqscreen.fundamentals import BudgetExhausted, RequestBudget

    b = RequestBudget(2)
    b.take()
    b.take()
    with pytest.raises(BudgetExhausted):
        b.take()
    unlimited = RequestBudget(None)
    for _ in range(100):
        unlimited.take()
    assert unlimited.used == 100


# --------------------------------------------------------------------------- empty responses (audit P2)
def _empty_provider(monkeypatch, f):
    monkeypatch.setitem(f.FETCHERS, "yfinance", (lambda tk, ctx: (pd.DataFrame(columns=f.FUND_COLS), 3), None))


def test_every_provider_response_empty_does_not_crash(fresh_demo_db, monkeypatch):
    from eqscreen import fundamentals as f
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    cfg["fundamentals"]["source"] = "yfinance"
    _empty_provider(monkeypatch, f)
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    res = f.fetch_fundamentals(cfg, ["SYN001", "SYN002", "SYN003"], conn, "empty-1", get_logger())
    assert res["rows"] == 0 and res["empty"] == 3 and res["ok"] == 0
    assert conn.execute("SELECT COUNT(*) FROM fundamentals WHERE source = 'yfinance'").fetchone()[0] == 0
    log = pd.read_sql_query("SELECT status FROM fetch_log WHERE run_id = 'empty-1'", conn)
    assert (log["status"] == "empty").sum() == 3


def test_empty_cached_files_are_reused_without_crashing(fresh_demo_db, monkeypatch):
    from eqscreen import fundamentals as f
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    cfg["fundamentals"]["source"] = "yfinance"
    _empty_provider(monkeypatch, f)
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    f.fetch_fundamentals(cfg, ["SYN001", "SYN002"], conn, "empty-a", get_logger())     # writes empty caches
    calls = []
    monkeypatch.setitem(f.FETCHERS, "yfinance", (lambda tk, ctx: calls.append(tk) or (pd.DataFrame(), 3), None))
    res = f.fetch_fundamentals(cfg, ["SYN001", "SYN002"], conn, "empty-b", get_logger())
    assert res["cached"] == 2 and res["rows"] == 0 and calls == []        # reused, no provider calls


def test_zero_byte_cache_file_is_refetched(fresh_demo_db, monkeypatch):
    from pathlib import Path
    from eqscreen import fundamentals as f
    from eqscreen.logging_utils import get_logger

    cfg, conn = fresh_demo_db
    cfg["fundamentals"]["source"] = "yfinance"
    cache = Path(cfg["project"]["raw_dir"]) / "fundamentals" / "yfinance" / "SYN001.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"")                                                 # interrupted write
    good = f.parse_fmp([{"date": "2023-12-31", "filingDate": "2024-02-15", "revenue": 1, "netIncome": 1}], [], "SYN001", 90)
    good["source"] = "yfinance"
    monkeypatch.setitem(f.FETCHERS, "yfinance", (lambda tk, ctx: (good, 3), None))
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    res = f.fetch_fundamentals(cfg, ["SYN001"], conn, "zb", get_logger())
    assert res["ok"] == 1 and res["cached"] == 0 and cache.stat().st_size > 0
