# Factor and ratio definitions

All inputs are point-in-time: at each month-end the screen uses the latest annual
statement whose `available_date` is on or before that date (see
[methodology.md](methodology.md#point-in-time-data)). Prices are month-end closes.

## Inputs and adjustments

| Symbol | Definition | Notes |
|---|---|---|
| Price | Month-end `Close` (split-adjusted, not dividend-adjusted) | Used for multiples. Dividend adjustment would understate historical market caps. |
| Total-return price | Month-end `Adj Close` (split- and dividend-adjusted) | Used for momentum and all returns. |
| Shares | Diluted weighted-average shares × split factor | Split factor = product of split ratios dated after the statement's filing date, so as-reported shares match the split-adjusted price. Set per source (`fundamentals.split_adjust`): a provider that already restates history must not be adjusted twice. |
| EPS | Diluted EPS ÷ split factor | If missing (Alpha Vantage), derived as net income ÷ shares. |
| Market cap | Price × shares | Invariant: raw price × as-reported shares ≈ split-adjusted price × split-adjusted shares. Dividend-adjusted prices are never used here. |
| EV | Market cap + total debt − cash | A missing debt or cash fact is not treated as zero: EV and every EV-based multiple and factor are left blank (`factors.ev_missing_policy: exclude`). The stock is still scored on earnings yield and book/market. `ev_coverage.csv` and the `ev_input_coverage` check report how many stocks are affected. `assume_zero` restores the old shortcut. Minority interest and preferred stock are not included. |
| EBIT | Operating income (EBIT where the provider reports it) | |
| EBITDA | Provider EBITDA, else EBIT + depreciation & amortization | |

## Valuation multiples (shown in the screen)

| Ratio | Formula | When undefined |
|---|---|---|
| P/E | Price ÷ EPS | EPS ≤ 0 |
| P/B | Market cap ÷ book equity | Equity ≤ 0 |
| EV/EBITDA | EV ÷ EBITDA | EV ≤ 0 or EBITDA ≤ 0; Financials |
| EV/EBIT | EV ÷ EBIT | EV ≤ 0 or EBIT ≤ 0; Financials |
| PEG | P/E ÷ (EPS growth × 100) | P/E undefined, growth ≤ 0, or the prior statement is not about one year earlier |

Multiples are for reading. Ranking uses their inverses (yields), because a multiple
blows up as earnings approach zero and has no sensible value when earnings are
negative, while a yield passes smoothly through zero.

## Scored factors

Higher is more attractive for every factor.

| Factor | Formula | Group | Rationale |
|---|---|---|---|
| Earnings yield | EPS ÷ price | Value | Inverse P/E. Negative earnings give a negative yield and rank at the bottom. |
| EBIT / EV | EBIT ÷ EV | Value | Capital-structure-neutral earnings yield; the "earnings yield" in Greenblatt's Magic Formula. |
| EBITDA / EV | EBITDA ÷ EV | Value | As above, before depreciation, so less sensitive to capital intensity. |
| Book / market | Book equity ÷ market cap | Value | The classic Fama-French value measure. |
| Momentum 12-1 | Total return from month t−12 to t−1 | Momentum | Skips the latest month, which tends to reverse (short-term reversal). Jegadeesh & Titman (1993). |
| ROE | Net income ÷ book equity | Quality | Off by default (`factors.weights.quality: 0`); always reported for analysis. |

EV-based factors are blank for **Financials** (`factors.ev_excluded_sectors`): bank
debt is operating inventory, not financing, so EV and EBIT are not meaningful.
Financials are still scored on earnings yield and book/market.

## From raw factors to a score

For each month separately:

1. **Winsorize** each raw factor at the 1st and 99th percentile, so one data error or
   extreme value cannot dominate.
2. **Z-score** it across stocks (mean 0, standard deviation 1).
3. **Sector-neutralize** (`factors.sector_neutral: true`): subtract the sector average
   z-score (sectors with at least 3 names), then re-standardize. Without this, "cheap"
   mostly means "in a structurally low-multiple sector" such as Energy or Utilities.
4. **Group scores**: the value score is the mean of the available value z-scores
   (at least `min_value_subfactors`, default 2), re-standardized. Momentum and quality
   scores are their single z-scores.
5. **Composite** = weighted average of group scores (default 50% value, 50% momentum).
   A stock needs every positively weighted group to receive a composite.
6. Months with fewer than `min_names_per_date` composites (default 30) are not scored.

## Earnings quality and cash flow

Full meaning, edge cases and limits of each indicator: [earnings_quality.md](earnings_quality.md).

| Measure | Formula | Use |
|---|---|---|
| Free cash flow | Operating cash flow − capex | Shown in screens, notes and the DCF base year |
| FCF yield | FCF ÷ market cap | Shown and given an IC diagnostic; not in the composite |
| FCF margin | FCF ÷ revenue | Shown |
| Cash conversion | Operating cash flow ÷ net income | Shown; blank when net income ≤ 0 |
| Accruals ratio | (Net income − operating cash flow) ÷ average total assets | Scored as "low accruals" (the negative), group `earnings_quality`, weight 0 by default |

The accruals ratio measures how much of reported profit is not yet cash. Sloan (1996)
found that firms with high accruals tended to have weaker subsequent earnings and returns.
Low or negative values mean earnings are backed by operating cash. It can mislead:

- **Fast growers** build receivables and inventory, which raises accruals without any
  quality problem.
- **Acquisitions, divestitures and one-off items** move net income or assets in ways unrelated to
  earnings quality.
- **Financials** have operating cash flows dominated by lending and deposits; read the
  ratio with care for banks and insurers.
- It needs two consecutive balance sheets (average assets), so the first year of a
  company's history has no value.

Capex is stored as a positive amount spent; providers that report it as a negative outflow
(yfinance, FMP) are sign-flipped on parsing.

## Edge cases and missing data

| Factor | Edge case | Handling |
|---|---|---|
| Earnings yield | Negative or near-zero EPS | Yield stays defined and ranks loss-makers last; P/E is left blank. |
| EBIT / EV, EBITDA / EV | Negative EV (cash above market cap plus debt), negative EBIT | Blank when EV ≤ 0. Negative EBIT gives a negative yield. Blank for Financials. |
| EBITDA / EV | EBITDA is not free cash flow | Ignores capex and working capital; capital-heavy firms look cheaper than their cash generation. Kept alongside EBIT/EV, which charges depreciation. |
| Book / market | Negative book equity (buybacks, accumulated losses) | Negative value ranks last; P/B is left blank. |
| ROE | Very small positive equity inflates ROE | Blank when equity ≤ 0; winsorization caps the rest. Off in the default composite. |
| Momentum 12-1 | Fewer than 12 months of prices (recent listings) | Blank, so the stock has no composite until it has a year of history. |
| PEG | Negative or zero growth; growth over a period that is not one year | Blank. PEG is shown for reading, never scored. |

**Thin sectors.** Sector neutralization subtracts the sector mean only when the sector has
at least 3 scored names that month; otherwise the market-wide z-score is kept.
Standardization is always market-wide, never within a sector, so a one- or two-stock
sector cannot produce a zero or unstable standard deviation. `_zscore` returns blank
when fewer than 3 values or zero dispersion exist.

**Missing factors.** Within value, the score averages the sub-factors that are available
(at least 2 of 4) and is re-standardized: missing sub-factors are reweighted, not
penalized. Across groups there is no reweighting: a stock missing any positively
weighted group (for example, no momentum yet) gets no composite. Financials lose the two
EV factors by design and are scored on earnings yield and book/market.

## Labels

| Label | Rule |
|---|---|
| Research candidate | Composite percentile above `screen.candidate_quantile` (top decile) |
| Lower priority | Composite percentile at or below `screen.lower_priority_quantile` (bottom decile) |
| Neutral | Everything in between |
| Not scored | Missing a required group score |

These are mechanical screening classifications that decide where an analyst looks first.
They are not investment recommendations, ratings or price targets.

## Rule-based baseline

A deliberately simple screen to benchmark the composite against:

1. Exclude Financials (`screen.rule_based.exclude_sectors`).
2. Keep stocks with positive 12-1 momentum and positive EBIT/EV.
3. Rank by EV/EBIT, lowest first. The live screen lists the top `top_n` (default 50);
   the backtest buys the cheapest decile of the eligible names.
