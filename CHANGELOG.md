# Changelog

## 0.7.6: sensitivity-grid settings

- **Sensitivity settings** (P3): a NaN or infinite step, a negative step count or a non-integer step
  count produced an empty or meaningless grid or a generic error. `wacc_step` and `growth_step` must
  now be finite and in (0, 0.05]; `steps` a whole number from 0 to 10. Checked with the other
  assumptions, so the DCF and research reports refuse with a clear message.
- Test count: 193 collected; with declared dependencies installed, all pass except 7 opt-in live checks,
  plus the real-SEC-fixture module (an 8th skip) until a fixture is captured.

## 0.7.5: non-finite inputs and invalid dates

- **Non-finite DCF inputs** (P2): NaN growth produced a NaN value, an explicit NaN beta produced a NaN
  WACC, and a NaN scenario was silently skipped by the weighted sum (a 21.83 base and an invalid bear
  at 50/50 gave a plausible-looking 10.92). Every assumption and override must now be a finite real
  number; non-finite WACC or value per share raise; `weighted_value` refuses any invalid scenario value.
- **NaT dates** (P3): `as_of: NaT` and `prepared_on: NaT` parsed without error. Dates are now checked
  after parsing; NaT-like values are refused.
- README limitations now state that the ML comparison uses only stocks with observable 12-month
  outcomes and that no result is live-tradable performance.
- Test count: 182 collected; with declared dependencies installed, all pass except 7 opt-in live checks,
  plus the real-SEC-fixture module (an 8th skip) until a fixture is captured.

## 0.7.4: valuation input validation

- **Scenario probabilities** (P2): weights summing to 1 could still be negative or above 1 (e.g. bear
  -25%, base 125%), producing a weighted value outside the scenario range. Each weight must now be a
  finite number between 0 and 1.
- **Assumption metadata** (P2/P3): `ticker` and `as_of` are mandatory in assumption files (standalone
  DCF and research reports refuse or omit files without them); malformed dates are refused; files
  without `prepared_on` are labelled undated in both reports and flagged.
- **Run summary** (P3): `filing_dates_estimated_share` is computed for the current universe only.
- Test count: 165 collected; with declared dependencies installed, 158 pass and 7 opt-in live checks
  skip, plus the real-SEC-fixture module (an 8th skip) until a fixture is captured.

## 0.7.3: ingestion resilience and integrity scope

- **All-empty provider responses** (P2): ingestion called `pd.concat` with no objects when every
  ticker returned an empty frame. Empty frames are now filtered by row count; zero-byte or
  truncated cache files are treated as missing and refetched. Regression tests: every response
  empty, empty caches reused, zero-byte cache refetched.
- **Integrity scope** (P3): blocking checks (`data_integrity`) cover the current universe and the
  benchmark; problems in records retained for other tickers are a `retained_records_integrity`
  warning instead of an assertion failure.
- **No `assert` for correctness**: point-in-time, purge-gap, duplicate-key and universe checks now
  raise explicit errors, which `python -O` does not strip; tested in a `-O` subprocess, plus a test
  that rejects `assert` statements in `src/`.
- Test count: 153 collected; with declared dependencies installed, 146 pass and 7 opt-in live checks
  skip, plus the real-SEC-fixture module (an 8th skip) until a fixture is captured.

## 0.7.2: monitoring and recovery fixes

- **Quality checks scoped to the current universe** (P1): `fundamental_coverage` counted statements
  of tickers no longer in the universe (it could report 60/5 covered), and `price_staleness_days`
  used the newest price of any stored ticker. All checks now use only current securities; the
  coverage detail reports how many outside tickers were ignored. New `stale_tickers` warning
  (per-ticker last price) so one fresh ticker cannot hide many stale ones; the benchmark check also
  flags a benchmark that ends before the universe. The validation report is scoped the same way.
- **Retry status** (P2): a stage that failed and then succeeded under the same run id left the run
  marked failed. Every stage execution is now recorded in `pipeline_stage_runs`, and the run's
  status and message come from the latest attempt of each stage.
- Test count for this version: 145 collected; with the declared dependencies installed, 138 pass and
  7 opt-in live checks skip, plus the real-SEC-fixture module (reported as an 8th skip, with no
  collected tests) until a fixture is captured. Without the `markdown` package, the 5 HTML/report
  tests fail, as they should: it is a declared dependency.

## 0.7.1: point-in-time follow-ups

- **Standalone DCF date and ticker checks** (P1): `dcf.run` / `python -m eqscreen.dcf` refuse an
  assumptions file for another ticker, a requested `--as-of` that differs from the file's month,
  and a month with no statement-backed base. Previously an explicit `--as-of` could value June 2016
  with assumptions dated December 2025.
- **Retrospective labels** (P2): key risks signed off, or DCF assumptions written (`prepared_on`,
  now recorded by `--init`), more than `research.contemporaneous_days` (default 45) after the report
  month-end are labelled retrospective in the report banner and section headings, and returned in
  the report metadata. The research report also omits a DCF written for another ticker.

## 0.7.0: independent audit fixes

**Bugs fixed (all reproduced first, then covered by regression tests)**
- P1-A: a historical research report could include a DCF valued at a later date. A DCF is now
  included only when its assumptions are dated to the report month; otherwise the report says
  why it was omitted.
- P1-B: the backtest could skip calendar months with no usable signal and then annualize the
  gapped series. Returns now run on an unbroken monthly calendar; rebalances are scheduled by
  elapsed months, missed rebalances carry the holdings (counted as `carried_rebalances`), and
  performance statistics refuse non-consecutive months.
- P2-C: switching to a CSV universe, or a failed Wikipedia change-table parse, could keep stale
  index membership. Universe builds now clear membership before rebuilding it.
- P2-D: `max_requests_per_run` was checked between tickers, so one ticker could exceed it. It is
  now a hard budget on every HTTP attempt, retries included.

**Research quality**
- DCF assumptions carry a `sources:` block, printed in the DCF and research reports; blanks flagged.
- Guards against pathological DCF inputs and non-positive base values.
- Key risks need real text and a dated "Reviewed by" sign-off to remove DRAFT; uncited risks flagged.

**Engineering**
- Shared run id across Airflow tasks (`run(..., run_id=)`, `--run-id`), one lineage row per DAG run.
- GitHub Actions CI: pinned Python 3.12 test suite, demo and validation, Docker build and run.

## 0.6.0: review follow-ups

- **DCF guards and tests**: explicit tests for terminal growth equal to and above WACC, extreme
  forecast growth, sensitivity cells in the undefined zone, and percentages typed as whole
  numbers; a note when terminal growth exceeds the risk-free rate.
- **Reverse DCF**: the terminal growth and the constant revenue growth implied by the current
  price (bisection, round-trip tested; reports "outside the searched range" rather than a number
  when no solution exists). In the DCF and research reports.
- **Shareable reports**: research, DCF and validation reports are also written as self-contained,
  print-ready HTML; `research --pdf` writes a PDF via Playwright (optional extra `[pdf]`).
- **Console**: one line per stage and a results summary by default; `--verbose` or
  `logging.console: verbose` for the full log. The log file is unchanged.
- **Docker**: `Dockerfile` and `.dockerignore` pinning the tested versions (not built in the
  development sandbox).
- **Notebook** now walks through survivorship, earnings quality, DCF scenarios and reverse DCF,
  the research report and the validation report.
- Backtest results are described as illustrative throughout.

## 0.5.0: research report, validation and point-in-time membership

- **Historical S&P 500 membership** rebuilt from Wikipedia's dated change table; stocks are
  scored, and enter the equal-weight benchmark, only in months they were members. Survivorship
  disclosure (earliest recorded change, departures in the test period, how many lack data) in
  the run summary, HTML report, research report and validation report.
- **Earnings-quality trends** (`eqscreen.earnings_quality`): multi-year accruals, cash conversion
  and FCF history (point in time), FCF CAGR, universe percentile of accruals, and mechanical
  flags; full write-up of each indicator in `docs/earnings_quality.md`.
- **DCF scenarios**: bear/base/bull overrides of any assumption, optional probability weights,
  scenario table and CSV.
- **Research report** (`python -m eqscreen.research`): fundamentals, peer multiples, DCF scenarios,
  earnings quality, factor rankings, price risk, key risks (analyst file + data flags), source
  filings with EDGAR links, assumptions and limitations; DRAFT until key risks are written.
- **Validation report** (`python -m eqscreen.validate`): filing-availability checks, point-in-time
  check, corporate actions, quality checks, fetch log, survivorship, real vs synthetic comparison
  with generated observations; `--archive` preserves the inputs with a database hash.

## 0.4.0: intrinsic value and earnings quality

- **Cash-flow data** from all four providers: operating cash flow, capex (stored as a positive
  amount spent) and total assets. Old databases gain the columns automatically.
- **Earnings quality**: accruals ratio (point in time, average of two balance sheets), cash
  conversion, FCF, FCF yield and FCF margin for every stock; "low accruals" as an optional
  composite group (weight 0) with IC diagnostics; shown in screens and analyst notes.
- **DCF module** (`python -m eqscreen.dcf`): FCFF model with WACC build, equity bridge and a
  WACC × terminal-growth sensitivity grid. Base year from point-in-time filings; forward-looking
  assumptions must be set by the analyst (`--init` writes a template with history for
  reference and refuses to run while any is blank). Tested against the Gordon-growth
  perpetuity identity. Documented in `docs/dcf.md`.
- **Reproducibility**: `requirements-tested.txt` pins the verified package versions; each
  run's `run_summary.json` includes a manifest of versions, configuration hash and data ranges.
- **README**: requirements and a table of credentials and environment variables.
- The shuffled-score test is now a 20-permutation test; the single-shuffle version was
  fragile to random noise (no code defect).

## 0.3.0: source-code audit fixes

**Bugs fixed**
- Max drawdown ignored the starting NAV, so a loss in the first month(s) reported 0%.
  Now measured from a peak of at least 1.0; regression tests start with a loss.
- The second (cached) monthly run of a non-SEC source crashed when a stored period had a
  blank field (found by the new fetch-loop test).
- Decile `n_dates` counted all dates for every decile; now per decile.

**Financial validity**
- Missing debt or cash is no longer treated as zero. EV and EV-based factors are blank
  unless `factors.ev_missing_policy: assume_zero`; `ev_coverage.csv` and an
  `ev_input_coverage` quality check report how many stocks are affected.

**Backtest**
- Headline now trades one month after the signal (`execution_lag_months: 1`); same-close
  trading is reported as `composite_lag0`, labelled optimistic.
- Missing-return sensitivity: `composite_missing_loss` treats each missing month as a total
  loss; `missing_return_share` reports coverage.
- Model vs composite comparison uses exactly the same stock-months (`composite_same_cohort`).
- Report and README describe results as a retrospective research diagnostic.

**Audit trail**
- Every HTTP request and retry to FMP, Alpha Vantage, SEC and Wikipedia is logged with
  endpoint, status, attempt and UTC time; API keys are never logged. Docs describe the
  logging granularity exactly.

**Real-data evidence and analyst output**
- `scripts/capture_sec_fixture.py` saves a trimmed real SEC filing history;
  `tests/test_real_fixture.py` checks originals vs restatements, accession links and
  share continuity across documented splits (skipped until a fixture is captured).
- `python -m eqscreen.note --ticker X`: one-page analyst note from the pipeline's own
  point-in-time data, with EDGAR filing links for the SEC source.
- `assets/demo_report_synthetic.png`: report screenshot, labelled synthetic.

## 0.2.0: methodology review fixes

**Point-in-time correctness**
- Fundamentals are versioned: one row per (period, filing that changed it), with
  `version_type` (original / amendment / restatement / revision_observed) and SEC accession.
- SEC parser replays filings in date order: originals are used until a 10-K/A or a
  restated comparative is filed, the revision only after its filing date.
- yfinance / FMP / Alpha Vantage: changed values between downloads are stored as new
  `revision_observed` versions instead of overwriting history.
- The SQL join picks the latest *period*, then the latest *version* known at each
  month-end (previously it ordered by filing date first, so an amendment to an old year
  could displace a newer statement). Prior-year figures for growth are point-in-time too.
- Staleness is measured from the fiscal period end.
- Old databases are migrated automatically.

**Backtest**
- `execution_lag_months` option; every run also reports a one-month-delay robustness result.
- Newey-West (HAC) t-statistics for excess returns and ICs; naive versions kept for comparison.
- Average number of holdings reported; `universe_breadth` warns when a "top decile" is too few names.

**Corporate actions**
- `split_adjust` is per data source.
- Live test (`EQSCREEN_LIVE=1`) checks share-count continuity across the AAPL 2020,
  TSLA 2022 and NVDA 2024 splits and says which way to fix the setting.

**Presentation**
- Labels renamed to Research candidate / Neutral / Lower priority, described as mechanical
  screening classifications.
- Framed as a current-universe historical factor study; README and methodology separate
  what unit tests, the synthetic test, live checks and real-data backtests each establish.
- Factor edge cases, thin-sector handling, missing-factor policy and an ML leakage audit documented.

**Tests added:** amended-filing (reviewer's suggestion), observed revisions, SEC versioning,
execution lag, Newey-West, universe breadth, live corporate actions.

## 0.1.0
Initial pipeline: universe, prices, fundamentals (4 sources), quality checks, factors,
screens, backtest, purged-CV model check, HTML report, scheduling.
