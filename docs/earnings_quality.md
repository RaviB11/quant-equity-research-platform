# Earnings quality

Three indicators ask one question from different angles: **is reported profit turning
into cash?** They are computed for every stock each month (screens, factor diagnostics)
and as multi-year histories in research reports and analyst notes
(`eqscreen.earnings_quality`). Every figure is point in time: each fiscal year uses the
latest version of its statement published by the month in question.

Inputs come from the statements: net income, operating cash flow (OCF), capital
expenditure (capex, stored as a positive amount spent), revenue and total assets.

## 1. Accruals ratio

**Formula:** (net income − operating cash flow) ÷ average total assets (this year's and
last year's balance sheet).

**Meaning.** Accruals are the part of profit recognized before cash arrives (or after it
leaves): rising receivables, inventory build-ups, deferred costs, falling payables. Scaling
by assets makes companies comparable. Sloan (1996) found that firms with high accruals
tended to have less persistent earnings and lower subsequent returns, because accrual-heavy
profit is more likely to reverse.

**How to read it.** Negative or low: profit is backed by cash (typical for mature companies,
whose depreciation lifts OCF above net income). High and rising: more of the profit depends
on accounting estimates; look at working-capital lines in the cash-flow statement. The screen
compares each company with the universe that month; research reports flag the highest 20%.

**Edge cases.**
- Needs two consecutive annual balance sheets, so a company's first year has no value.
- Loss-making companies: still defined (net income can be negative), but a loss with strong
  OCF gives a very negative ratio that signals depreciation or impairments, not quality.
- Assets that change sharply (an acquisition, a spin-off) distort the average.

**Limits.**
- **Fast growers** build receivables and inventory to support growth, which raises accruals
  with no quality problem. Compare with revenue growth before drawing conclusions.
- **One-off items** (impairments, gains on sales, restructuring) move net income without cash.
- **Financials:** bank and insurer operating cash flow is dominated by lending, deposits and
  investment flows, so the ratio says little.
- The cash-flow statement itself can be shaped by timing (delaying payables, factoring
  receivables), so low accruals are not proof of quality.

## 2. Cash conversion (operating cash flow ÷ net income)

**Formula:** OCF ÷ net income, only when net income is positive.

**Meaning.** How many dollars of operating cash each dollar of reported profit produced.
Above 1.0 is common for asset-heavy, mature companies (depreciation is a non-cash charge);
persistently below 1.0 means profit is running ahead of cash.

**How to read it.** Look at the trend, not one year. Research reports flag when it is below
the configured floor (`research.flags.min_cash_conversion`, default 0.8, a threshold chosen
by this project) in two of the last three years.

**Edge cases.**
- Undefined (blank) when net income is zero or negative: the ratio flips sign and loses meaning.
- Near-zero net income produces extreme values; read the underlying amounts.

**Limits.**
- Strongly affected by depreciation intensity, so compare within an industry, not across.
- Working-capital swings make single years noisy; a one-year dip can be a timing effect.
- Stock-based compensation is added back in OCF, flattering cash conversion for companies
  that pay heavily in stock.

## 3. Free cash flow and its trend

**Formulas:** FCF = OCF − capex. FCF margin = FCF ÷ revenue. FCF yield = FCF ÷ market cap.
FCF growth = year-on-year change, only when both years are positive. Research reports also
show a compound annual growth rate over the history window when the first and last years
are both positive.

**Meaning.** Cash left after maintaining and expanding the asset base: the cash available
for dividends, buybacks, debt repayment or acquisitions, and the quantity a DCF values.

**How to read it.** A steady or rising FCF margin alongside rising earnings supports earnings
quality. Earnings rising while FCF falls is a warning worth explaining. Research reports flag
FCF falling in each of the last three years (`fcf_decline_years`) and negative FCF in the latest
year.

**Edge cases.**
- Growth rates are not computed across a sign change (negative to positive is not a percentage).
- Capex here is purchases of property, plant and equipment; acquisitions and capitalized
  software are not included unless the provider includes them.

**Limits.**
- **Investment cycles:** a company in a heavy build-out phase shows low or negative FCF by
  design; that can be value-creating.
- **Lumpy capex** makes annual FCF volatile; the multi-year view matters more than one year.
- FCF before interest-related adjustments: it is not identical to the unlevered FCFF used in
  the DCF (which starts from EBIT × (1 − tax)).
- Leases: capitalized leases change where payments appear between operating and financing
  cash flow across reporting standards and years.

## Where these appear

| Place | What is shown |
|---|---|
| `screen_composite_latest.csv` | FCF yield, accruals ratio, cash conversion for every stock |
| Factor diagnostics (`ic_summary.csv`, report) | IC of low accruals and FCF yield vs forward returns |
| Composite | Low accruals as group `earnings_quality`, weight 0 by default |
| Analyst note, research report | Multi-year history table, trend statistics and flags |
| DCF `reference` block | Historical OCF ÷ net income, capex and D&A ratios |
