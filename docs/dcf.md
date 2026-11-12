# DCF valuation

`python -m eqscreen.dcf --ticker AAPL` values one company by discounting its unlevered free
cash flow. It is a transparent, assumption-driven model meant to sit next to the screen,
not a replacement for analyst judgment.

## Workflow

1. Run the pipeline (live) so the company's statements and prices are in the database.
2. `python -m eqscreen.dcf --ticker AAPL --init` writes `config/dcf/AAPL.yaml`. It records
   the valuation month and fills a `reference` block with the company's recent history from
   its filings: revenue growth, EBIT margin, D&A and capex as a share of revenue, and
   operating cash flow to net income. Every forward-looking field is blank.
3. Fill in the assumptions (decimals: 0.05 = 5%) and the `sources:` block. The file's `as_of`
   month is the valuation month; research reports only include the DCF for that same month.
4. `python -m eqscreen.dcf --ticker AAPL` writes `outputs/dcf/AAPL_<yyyymm>.md` plus the
   projection and sensitivity grid as CSV.

The model will not run while any assumption is blank, and it never supplies a default for a
forward-looking input.

## Method

| Step | Formula |
|---|---|
| Revenue | Base-year revenue grown by your rate for each forecast year |
| EBIT | Revenue × EBIT margin (one value or a path) |
| NOPAT | EBIT × (1 − tax rate) |
| Free cash flow to the firm | NOPAT + D&A − capex − change in net working capital |
| Change in NWC | (Revenue_t − Revenue_t−1) × NWC % of incremental revenue |
| Cost of equity | Risk-free rate + beta × equity risk premium (CAPM) |
| WACC | E/(D+E) × cost of equity + D/(D+E) × cost of debt × (1 − tax) |
| Terminal value | FCFF_N × (1 + g) / (WACC − g), Gordon growth |
| Enterprise value | PV of forecast FCFF + PV of terminal value, end-of-year discounting |
| Equity value per share | (EV − total debt + cash) / diluted shares (split-adjusted) |

**Data-derived inputs**, as known at the valuation month: base-year revenue, EBIT, D&A
(EBITDA − EBIT), capex, total debt, cash, diluted shares, share price, market cap, and a raw
beta regressed on up to 60 months of monthly returns against the benchmark (at least 24).
E uses market cap; D uses book debt.

**Your inputs**: revenue growth by year, EBIT margin, tax rate, D&A and capex as a share of
revenue, NWC share of incremental revenue, terminal growth, risk-free rate, equity risk
premium, pre-tax cost of debt. Optional: `beta` (overrides the regression), `total_debt` and
`cash` (if the filing data lack them).

## Sourcing the assumptions

| Input | Where to get it |
|---|---|
| Risk-free rate | Current yield on a long-dated government bond in the valuation currency (for US companies, a 10-year Treasury yield from Treasury or Federal Reserve data) at the valuation date |
| Equity risk premium | A published estimate you can cite, for example the implied ERP series Aswath Damodaran publishes, or your firm's house view |
| Pre-tax cost of debt | Yield on the company's traded bonds, or interest expense ÷ average debt from the filings |
| Tax rate | Statutory rate, or the effective rate from the income statement (tax expense ÷ pre-tax income); blend toward statutory for the long run |
| Growth and margins | Company guidance, consensus, and the history in the `reference` block. Growth should fade toward the terminal rate |
| Terminal growth | At or below long-run nominal GDP growth of the company's markets; above that it implies the company eventually outgrows the economy |

## One company, one date

An assumptions file records the company (`ticker`), the valuation month (`as_of`) and the
day it was written (`prepared_on`, set by `--init`). `ticker` and `as_of` are mandatory: a file
without them is refused, because nothing would say what is valued or when. A file without
`prepared_on` is accepted but labelled **undated** in the DCF and research reports, since its
timing relative to the valuation date cannot be shown. Scenario weights, if given, must each be
a probability between 0 and 1 and must sum to 1. The model refuses to value a different
ticker with it, or to value it at a different month (`--as-of` must match the file; create a
new file with `--init --as-of YYYY-MM` for another date), and refuses if no statement-backed
base exists at that month. If `prepared_on` is more than `research.contemporaneous_days`
(default 45) after the valuation month-end, the report is labelled a **retrospective
valuation**: written with hindsight, so it may use information not available at the time.

## Recording sources

`--init` adds a `sources:` block with one field per input group (revenue growth, EBIT margin,
tax rate, reinvestment, terminal growth, risk-free rate, equity risk premium, cost of debt,
scenarios). Write where each number comes from and when, for example "10-year Treasury yield,
Federal Reserve H.15, 2025-12-31" or "management guidance, Q4 2025 earnings call". The DCF and
research reports print the table and mark blank entries **not recorded**.

## Guards and the reverse DCF

- **Non-finite inputs are refused**: every assumption (and any beta, debt or cash override) must be
  a finite real number, so NaN, infinity, text or true/false never reach the model; a non-finite
  WACC or value per share is an error; and a probability-weighted value is refused if any scenario
  value is invalid (a plain sum would silently skip it). Dates that parse to "not a time" (`NaT`,
  `nan`) are errors, not missing values.
- **Sensitivity-grid settings are checked** with the assumptions: `wacc_step` and `growth_step` must
  be finite numbers above 0 and at most 0.05; `steps` must be a whole number from 0 to 10.
- **Pathological inputs are refused**: revenue growth at or below −100% or above 300% a year,
  EBIT margins outside ±100%, reinvestment ratios outside 0–1, implausible terminal growth, and
  a base year with non-positive revenue, shares or price.

- **Terminal growth at or above WACC is refused**, and so is a spread under 0.5 percentage
  points: the Gordon formula is undefined or explosive there. Sensitivity cells in that zone
  show "n/a". Tests cover growth equal to and above WACC, extreme forecast growth (−30% to
  +100% a year) and percentages typed as whole numbers (4 instead of 0.04).
- **Terminal growth above the risk-free rate** triggers a note in the report; long-run growth
  above the risk-free rate is hard to justify for most companies.
- **Reverse DCF.** With every other base-case assumption held fixed, the model solves (by
  bisection) for the terminal growth that makes value per share equal the current price, and
  separately for a single constant revenue growth rate over the forecast years that does the
  same. If no value in the searched range works, it says so instead of returning a number.
  This calibrates your assumptions against the market without inventing inputs: if the price
  implies 5% terminal growth and you assume 2.5%, the gap is the debate.

## Sanity checks to read every time

- **Terminal share of EV.** The report states it. When it is large (often 70%+), the
  value mostly reflects the terminal assumptions, so read the sensitivity table.
- **Reinvestment and growth consistency.** Terminal growth needs reinvestment: if capex
  barely covers D&A and NWC is flat, high terminal growth is not funded.
- **WACC − g.** The model refuses a spread under 0.5 percentage points; even a 1–2 point
  spread makes the value very sensitive.
- **Beta.** A raw regression on a few years of monthly data is noisy; consider an
  industry or adjusted beta and set `beta` explicitly.

## Limits

- No adjustment for leases, minority interests, pensions, preferred stock, or options
  beyond diluted shares; equity value per share is approximate for companies where these
  are material.
- Annual base year; no mid-year discounting convention.
- Not suited to banks and insurers (FCFF and EV are not meaningful), or to companies with
  negative or highly cyclical earnings without explicit normalization.
- The output is a valuation under stated assumptions, not a price target or recommendation.
