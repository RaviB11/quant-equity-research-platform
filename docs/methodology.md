# Methodology

## Pipeline

```
universe ─► prices ─► fundamentals ─► quality ─► factors ─► screen ─► backtest ─► model ─► report
 Wikipedia   yfinance   yfinance/FMP/    checks,    SQL PIT     ranked    top-decile  purged   HTML +
 or CSV                 Alpha Vantage/   stop on    join,       lists     portfolios  CV       PNGs
                        SEC EDGAR        failure    z-scores
```

Universe builds clear stored index membership before rebuilding it, so switching to a CSV
universe (or a failed change-table parse) never reuses a previous run's membership. With
Airflow, all tasks of one DAG run share a run id and record into one `pipeline_runs` row. Each
stage execution is a row in `pipeline_stage_runs`; the run's status comes from the latest attempt
of each stage, so a task that fails and then succeeds on Airflow's retry leaves the run
successful, while a stage whose latest attempt failed keeps the run failed.

Every stage reads from and writes to one SQLite database (`data/equity_screener.db`),
so stages can be rerun independently (`--stages` or `--from-stage`). Demo mode swaps
the three fetch stages for a synthetic data generator and writes to a separate
database and output folder.

| Table | Contents |
|---|---|
| `securities` | Current universe with GICS sector and CIK |
| `prices_monthly` | Month-end `close` (split-adjusted) and `adj_close` (total return) |
| `splits` | Split date and ratio per ticker |
| `fundamentals` | Annual statements normalized across providers, one row per version, with `available_date`, `version_type` and SEC `accession` |
| `factor_panel` | Every stock-month: ratios, factor z-scores, composite, label, forward returns |
| `fetch_log` | UTC-stamped rows for every HTTP request and retry this code makes (endpoint, status, attempt; no API keys) and one outcome per ticker (ok, cached, empty, error, rate-limited, skipped). yfinance's internal requests are logged per ticker only. |
| `quality_checks` | Each check's result per run |
| `pipeline_runs` | Run id, stages, start/finish time, status derived from the latest attempt of each stage |
| `pipeline_stage_runs` | Every stage execution (attempt number, times, status, error), so retries and failures stay visible |

## Point-in-time data

Lookahead bias, using information that was not public yet, is the most common way a
backtest flatters itself. A point-in-time *join* is not enough on its own: the
*values* must also be the ones known at the time. A company can report FY2020 in
early 2021 and restate it in 2023; a 2021 screening decision must not see the 2023
figures. The pipeline separates five things:

| Concept | Where it lives |
|---|---|
| Reporting period | `period_end` |
| Original filing | the `version_type = 'original'` row for that period |
| Amended or restated filing | later rows: `amendment` (10-K/A), `restatement` (changed comparative in a later 10-K), `revision_observed` (see below) |
| Availability date | `available_date` of each version: filing date, or estimate, or first observation |
| Value used at a screening date | the latest version of the latest period with `available_date <= month_end` (`sql/factor_inputs.sql`) |

**SEC EDGAR (`sec`) gives true filing-time history.** Each companyfacts record carries
the filing date and accession of the 10-K or 10-K/A that reported it. For every fiscal
year the parser replays the filings in date order and emits a new version only when a
filing changes what was known. The original 10-K is used until a later filing revises
it, and the revision is used only from its filing date. Each version keeps its
accession number, which links back to the filing on EDGAR. Caveat: companyfacts is a
current snapshot of XBRL data, and facts filed before XBRL tagging (roughly 2009–2011,
depending on filer size) are absent, so the earliest years may be missing or sparse.

**Other providers give today's figures.** yfinance, FMP and Alpha Vantage return the
latest values, which may already include restatements. The first download therefore
cannot be unwound, and that is a stated limitation. From then on, each run compares the
download with what is stored: if a period's values changed, the new values are added as
a `revision_observed` version dated the day they were first seen, instead of
overwriting history. Monthly runs build a genuine as-observed archive going forward.

**Availability dates.** The real filing date where the provider gives one (FMP's
`filingDate`, SEC's `filed`), otherwise fiscal year-end + `filing_lag_days` (default
90; large accelerated filers must file a 10-K within 60 days of year-end, smaller filers
within 75 or 90). Estimated dates are flagged in `available_date_is_estimated` and their
share is reported.

**Staleness.** A statement whose fiscal period ended more than `max_staleness_months`
before the screening date is ignored, measured from `period_end` so a late restatement
of an old year cannot make it look current.

**Tests.** `test_no_lookahead_scores_unchanged_when_future_removed` deletes everything
after a cutoff and checks every earlier score is identical.
`test_amended_filing_only_affects_scores_after_its_filing_date` inserts a 10-K/A that
multiplies EPS by five and checks that all scores before its filing date are unchanged,
that the amended EPS is used from its filing date, and that it later feeds growth as the
prior year. The loader also asserts that neither the current nor the prior-year
statement was published after the screening date.

Forward returns are stored in `factor_panel` for evaluation. No scoring function
reads them.

## Historical index membership

With `universe.source: sp500_wikipedia`, the universe stage also parses Wikipedia's dated
table of S&P 500 additions and removals and rebuilds membership spans by walking back from
today's list (`universe.reconstruct_membership`): a stock added on date *d* is a member from
*d*; one removed on *d* was a member until the day before. Stocks are scored, and enter the
equal-weight benchmark, only in months they were members (`universe.point_in_time_membership`).

What this fixes: the screen can no longer pick a company before it joined the index, which
matters because companies are often added after strong performance. What it does not fix:
companies that left the index are mostly missing from free price data (Yahoo usually drops
delisted tickers, and a reused ticker may now belong to a different company, so removed
tickers are deliberately not fetched). Their returns, often poor, are absent, and results
remain biased upward. Wikipedia's change history is also incomplete for early years;
membership before its earliest recorded change is assumed. Every run reports the earliest
recorded change, the number of departures in the test period and how many lack data
(`run_summary.json`, report, validation report).

## Data sources and their limits

| Source | Key | Filing dates | Notes |
|---|---|---|---|
| yfinance (default) | none | No (lag estimate) | Unofficial Yahoo endpoint. Only the last few annual statements are available, which limits the backtest window. |
| Financial Modeling Prep | `FMP_API_KEY` | Yes | Free tier is rate- and symbol-limited; check the current plan. Supports the `stable` and legacy v3 URLs (`fundamentals.fmp_base_url`). |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` | No (lag estimate) | Free tier allows very few calls per day; set `max_requests_per_run` and let the cache build up over several runs. |
| SEC EDGAR XBRL | none (requires a contact `User-Agent`) | Yes | Free, long history, true point-in-time. XBRL tags vary by company, so coverage of some fields is uneven. |

Per-ticker downloads are cached in `data_raw/fundamentals/<source>/` and reused for
`cache_days`. `max_requests_per_run` is a hard budget on HTTP attempts, retries included
(yfinance's three statement requests per ticker are charged too); when it runs out, the run
stops calling the provider and uses cached data for the remaining tickers. If a provider signals a rate limit, the run stops calling it, uses
cached data for the remaining tickers, and the coverage check reports the gap.

## Split adjustment

Yahoo restates `Close` for every split up to today. SEC filings report shares and EPS in
the units of their filing date. Without adjustment, a stock that split 4-for-1 after a
filing would show a market cap a quarter of its true size for all earlier months. The
pipeline multiplies as-reported shares (and divides EPS) by the product of split ratios
dated after the filing date, so

    raw price × as-reported shares ≈ split-adjusted price × split-adjusted shares.

Valuation always uses the split-adjusted (not dividend-adjusted) `Close`; dividends enter
only through `Adj Close` in returns.

The trap is double adjustment: some providers already restate past share counts for
later splits. `fundamentals.split_adjust` is therefore set per source, and two checks
catch a wrong setting:

- `share_count_discontinuity` (quality stage) warns when adjusted shares jump more than
  1.6x between consecutive annual reports.
- `tests/test_live.py` (run with `EQSCREEN_LIVE=1`) fetches AAPL, TSLA and NVDA, whose
  splits are well documented (4-for-1 in August 2020, 3-for-1 in August 2022, 10-for-1 in
  June 2024), and checks the share count is continuous across each split. Its failure
  message says which way to change the setting.

## Backtest

- **Formation:** at month-end *t*, rank stocks on the score known at *t* and buy the
  top decile (`top_quantile`, at least `min_holdings` names) in equal weight. On a small
  universe `min_holdings` binds and the portfolio is a larger slice than a decile; the
  summary reports the actual average number of names, and the `universe_breadth` check
  warns.
- **Execution timing:** the headline (`execution_lag_months: 1`) trades one month after
  the signal month-end. Trading at the same close the score was computed from is the
  usual academic convention and optimistic, since the data arrive after the close; it
  is reported as `composite_lag0` for comparison. With monthly data the realistic
  next-session price is not available, so the truth sits between the two; a daily
  execution model is the natural next step. Returns always start after the signal date.
- **Holding:** hold over (*t*, *t+H*]. Weights drift with returns inside the holding
  period. H = 1, 6 and 12 months are reported; the first is the headline.
- **Unbroken calendar.** The return series runs month by month from the first month with a
  usable signal to the last month with returns, and never skips a calendar month. Rebalances
  are scheduled by elapsed months (every H months from the start), not by counting months
  that happen to have signals. If a scheduled month has too few scored stocks, the current
  holdings are carried to the next scheduled rebalance and the month is counted as
  `carried_rebalances` in the summary. Performance statistics refuse a series whose months
  are not consecutive, so a gapped sample can never be annualized as if continuous.
- **Costs:** `cost_bps` per unit of traded weight at each rebalance (buying 100% of a
  new portfolio from cash costs 1 × 10 bps). No market impact, borrow or taxes.
- **Missing returns:** a holding with no return in a month (data gap, delisting) is
  treated as cash for that month and counted in `n_missing`. Because a delisting can
  mean a large loss, `composite_missing_loss` reruns the headline with every missing
  month as a total loss, and the summary reports `missing_return_share`, the share of
  holding-months affected. The gap between the two shows how much the cash assumption
  matters.
- **Benchmarks:** the equal-weight universe (same stocks, same months) isolates the
  effect of selection; SPY total return shows the cap-weighted market. Both use
  month-end adjusted closes on the same dates as the strategy, so return conventions
  match. Neither benchmark pays costs.
- **Longer holding periods change the schedule, not just the reporting.** With H = 6 or
  12 the portfolio is rebalanced only every H months and drifts in between
  (`test_buy_and_hold_weights_drift_within_holding_period`). Results then depend on the
  starting month ("rebalance timing luck"); averaging H staggered portfolios would remove
  that and is a natural extension.

### Metrics

| Metric | Definition |
|---|---|
| CAGR | (∏(1 + r))^(12/n) − 1 |
| Sharpe | mean(r − rf) ÷ sd(r − rf) × √12, rf from `risk_free_annual` |
| Excess return | mean monthly (strategy − equal-weight universe) × 12 |
| Information ratio | mean active return ÷ sd(active return) × √12 |
| t-stat of excess | Newey-West (HAC) t-statistic of the mean monthly active return, lags = ⌊4(n/100)^(2/9)⌋; the naive i.i.d. version is kept as `t_stat_excess_iid` for comparison |
| Max drawdown | worst peak-to-trough fall of the growth path |
| Rank IC | Spearman correlation of score with next-period return, per month |
| IC t-stat | Newey-West t-statistic of the mean IC with at least h − 1 lags, because h-month forward returns at consecutive months overlap |
| Decile spread | mean forward return by score decile, minus the month's cross-sectional mean |

## Machine-learning check (stretch goal)

Question: does learning the factor weights beat fixed equal weights out of sample?

- **Features:** the six factor z-scores (missing = 0, the cross-sectional mean).
- **Targets:** logistic regression on "12-month forward return above the month's
  median"; ridge regression on the 12-month forward return (winsorized 1/99 per month).
- **Validation:** `TimeSeriesSplit` over **dates**, not rows, with `gap = 12` months.
  A 12-month label formed at *t* uses returns through *t+12*, so without the gap the
  final training labels would overlap the test period. Each fold asserts this.
- **Comparison:** AUC and mean rank IC of each model against the fixed composite on
  the same test rows, plus a backtest of the model's out-of-sample score against the
  composite over the same months.
- **PCA** on the factor z-scores shows how much the value measures overlap.

### Leakage audit

| Step | Fitted on | Why it does not leak |
|---|---|---|
| Factor z-scores, winsorization, sector neutralization | One month's cross-section | Uses only that month's data, so no information crosses dates. |
| Missing-value imputation | Nothing (constant 0) | 0 is the cross-sectional mean of a z-score; no statistic is learned. |
| Label (above-median 12-month return) and target winsorization | One month's cross-section | Median and clip points come from that month only. |
| StandardScaler | Training fold only | Inside the scikit-learn pipeline, refit per fold. |
| Logistic / ridge coefficients | Training fold only | Refit per fold; test months are never seen. |
| Hyperparameters (C, alpha) | Not tuned | Fixed in config, so no selection on test data. A nested CV would be needed to tune them. |
| Train/test boundary | Dates, 12-month gap | Asserted per fold: the last training label ends before the first test month. |
| Full-sample coefficients and PCA | All data | Descriptive only, labeled as such, never used to score or backtest. |

Evaluation is by AUC and by mean rank IC per fold, so stability across periods is
visible fold by fold.

**Same cohort.** The model can only score stock-months that have a 12-month label, so
its backtest is compared with `composite_same_cohort`: the fixed composite restricted to
exactly the same (month, ticker) rows (`model.same_cohort_panel`). AUC and IC per fold
already use identical rows. Note that both are conditioned on the stock having 12 more
months of prices, a mild look-ahead in *who* is eligible, so compare the two with each
other, not with the main backtest. A model that cannot beat the fixed composite out of sample is a
legitimate result: it says the extra flexibility is not paying for itself.

If statement history is too short for at least two folds with `min_train_dates`
training months, the stage logs why and skips; the rest of the pipeline continues.

## Research report and validation

`python -m eqscreen.research --ticker X` assembles a one-company report from the pipeline's
own data, all as of one month. A DCF is included only when its assumptions file is dated to
the report month; a historical report never shows a later valuation, statement, price or
beta, and says why the DCF was omitted. The report stays DRAFT until the key-risks file has
real text and a "Reviewed by: name, YYYY-MM-DD" line; risks with no cited source (filing,
URL or "Source:") and DCF inputs with no recorded source are flagged. Analyst work dated well
after the report month (sign-off or DCF `prepared_on` more than `research.contemporaneous_days`,
default 45, after the month-end) is labelled **retrospective**: legitimate hindsight analysis,
but never presented as information available at the report date. A DCF written for another
ticker or another month is omitted. Contents: multi-year fundamentals (latest version of each year published by the report
date, split-consistent EPS), the nearest-market-cap sector peers, DCF scenarios from the
assumptions file, earnings-quality history and flags, factor rankings, price risk, the
analyst's key-risks file plus mechanical flags (thresholds in `research.flags`, described as
this project's choices), source filings with EDGAR links, assumptions and limitations.

`python -m eqscreen.validate` documents a live run: filing-lag distribution and late filers,
revision counts by type, accession coverage, a point-in-time check over every stock-month,
corporate-action and quality-check results, the fetch log, the survivorship disclosure, and
each headline metric side by side with the synthetic benchmark, with the reason the two can
differ. Observations about the real results are generated from the numbers only (for
example, whether an IC's Newey-West t exceeds 2). `--archive` zips the database, config,
DCF assumptions, universe and change lists and run manifest with the database's SHA-256.

## Intrinsic value (DCF)

A standalone, single-company valuation (`python -m eqscreen.dcf`); it does not feed the
screen or the backtest. The base year comes from the point-in-time statement version
public at the valuation month, and forward-looking assumptions are set by the analyst in
`config/dcf/<TICKER>.yaml`. Full method, sourcing guidance and sanity checks:
[dcf.md](dcf.md).

## Data quality checks

Every check describes the **current universe** (the `securities` table). The database can hold
prices and statements for tickers from earlier universes; those never count toward coverage or
freshness, and the coverage detail says how many outside tickers were ignored.

**Integrity scope.** `data_integrity` covers the records this run uses (current universe and
benchmark): duplicate securities, duplicate price rows or non-month-end price dates fail the run.
Records retained for tickers outside the universe are history the run does not read; problems
there appear as the `retained_records_integrity` warning (with the tickers named) rather than
stopping the run. Correctness guarantees (point-in-time, purge gap, key uniqueness) raise explicit
errors rather than using `assert`, so they also hold under `python -O`; a test enforces that no
`assert` statement guards behaviour in `src/`.

| Check | Fails when |
|---|---|
| `universe_size` | Fewer than `min_universe_size` securities |
| `price_coverage` | Share of tickers with 13+ months of prices below `min_price_coverage` |
| `price_positive` | Any non-positive or missing close |
| `price_staleness_days` | The universe's latest month-end is more than `max_price_staleness_days` before the last complete month (warning in demo mode) |
| `fundamental_coverage` | Share of tickers with fundamentals below `min_fundamental_coverage` |
| `available_after_period_end` | A statement is available before its period ends |

Warnings (logged and shown in the report, do not stop the run): more than
`max_stale_ticker_share` of universe tickers ending before the latest month (`stale_tickers`, so
one fresh ticker cannot hide many stale ones), extreme monthly returns, a missing or lagging
benchmark, share-count discontinuities, a universe too small for a real
top decile (`universe_breadth`), fewer than `min_ev_input_coverage` of non-financial
statements reporting both debt and cash (`ev_input_coverage`). Informational: share of estimated filing dates, number of
fiscal periods with more than one version (`revised_periods`). Code-level `assert`s guard invariants such as unique keys and month-end dates.

## Synthetic demo data

`--mode demo` builds a fake market so the pipeline can be tested offline. A latent
"cheapness" variable drives both reported valuations and next-month returns; no
momentum effect is planted; a few tickers split; filings lag year-end by 40 to 85 days.
The tests check that value factors recover the planted effect, momentum does not, and
shuffled scores show nothing. Demo results say nothing about real markets, and every
demo output is marked as synthetic.

## What the evidence shows

| Evidence | What it establishes | What it does not |
|---|---|---|
| Unit tests (`pytest`) | Components behave as specified on constructed inputs: parsers, point-in-time joins, split adjustment, backtest arithmetic, CV folds | That real provider data look like the fixtures |
| Synthetic planted-signal test | The pipeline recovers a known relationship (value) and does not invent one (momentum, shuffled scores) | That these factors predict real returns |
| Live data checks (`EQSCREEN_LIVE=1`) | Real price and filing data parse correctly and split handling holds on real corporate actions | Anything about returns |
| Historical evaluation on real data (`--mode live`) | How the method performed on the chosen universe and period, under the stated assumptions and biases | Future performance; results are inflated by survivorship bias |
| Paper-tracked screens going forward | Out-of-sample behavior after the method was fixed | (Not yet available: requires months of saved monthly screens) |

Saved monthly screens (`screen_composite_YYYYMM.csv`) are the raw material for the last
row: comparing each month's research candidates with what happened next, without
changing the method in between, is the only genuinely out-of-sample test.

## Known limitations

- **Survivorship bias.** This is a current-universe historical factor study: today's
  S&P 500 applied to the past. Companies removed from the index, acquired or bankrupt
  are missing, which overstates historical returns. Fixing this needs historical
  constituent lists and delisting returns.
- **Values from non-SEC sources are today's values.** Restatements before the first
  download are baked in (see Point-in-time data).
- **Short history from free sources.** yfinance gives only a few years of statements;
  use SEC EDGAR for a longer, point-in-time history.
- **Annual statements only.** Quarterly trailing-twelve-month figures would make the
  value factors more current.
- **Monthly execution model.** The headline trades a month late, same-close trading is
  optimistic; a daily model would sit between them.
- **Missing returns as cash.** See `composite_missing_loss` for the worst case.
- **Simplified EV.** No minority interest, preferred stock or leases.
- **Costs are flat.** Small, illiquid names would cost more to trade.
