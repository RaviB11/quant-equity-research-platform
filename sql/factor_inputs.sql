-- Point-in-time factor inputs over VERSIONED fundamentals.
--
-- For each stock and month-end:
--   1. keep every statement version whose available_date is on or before the month-end;
--   2. for each fiscal period, keep the latest of those versions (so an amendment is
--      used only after it was filed, and the original before that);
--   3. the most recent fiscal period is the "current" statement and the one before it
--      is the "previous" statement used for growth (both as known at that month-end).
-- A current statement whose period ended more than :max_staleness_days before the
-- month-end is ignored, so a company that stops reporting drops out instead of being
-- valued on stale numbers. The LEFT JOINs keep price rows with no usable fundamentals,
-- so price-only factors (momentum) still score.
--
-- Parameters: :source, :max_staleness_days

WITH known AS (
    SELECT
        p.ticker,
        p.month_end,
        f.period_end,
        f.available_date,
        f.available_date_is_estimated,
        f.version_type,
        f.accession,
        f.revenue, f.ebit, f.ebitda, f.net_income, f.eps_diluted, f.shares_diluted,
        f.total_equity, f.total_debt, f.cash, f.operating_cash_flow, f.capex, f.total_assets,
        ROW_NUMBER() OVER (
            PARTITION BY p.ticker, p.month_end, f.period_end
            ORDER BY f.available_date DESC
        ) AS version_rank
    FROM prices_monthly AS p
    JOIN securities AS s
      ON s.ticker = p.ticker
    JOIN fundamentals AS f
      ON f.ticker = p.ticker
     AND f.source = :source
     AND f.available_date <= p.month_end
),
periods AS (
    SELECT
        k.*,
        ROW_NUMBER() OVER (
            PARTITION BY k.ticker, k.month_end
            ORDER BY k.period_end DESC
        ) AS period_rank
    FROM known AS k
    WHERE k.version_rank = 1
)
SELECT
    p.ticker,
    p.month_end,
    p.close,
    p.adj_close,
    s.name,
    s.sector,
    cur.period_end,
    cur.available_date,
    cur.available_date_is_estimated,
    cur.version_type,
    cur.accession,
    cur.revenue,
    cur.ebit,
    cur.ebitda,
    cur.net_income,
    cur.eps_diluted,
    cur.shares_diluted,
    cur.total_equity,
    cur.total_debt,
    cur.cash,
    cur.operating_cash_flow,
    cur.capex,
    cur.total_assets,
    prev.total_assets   AS total_assets_prev,
    prev.eps_diluted    AS eps_prev,
    prev.net_income     AS net_income_prev,
    prev.shares_diluted AS shares_prev,
    prev.period_end     AS period_end_prev,
    prev.available_date AS available_date_prev
FROM prices_monthly AS p
JOIN securities AS s
  ON s.ticker = p.ticker
LEFT JOIN periods AS cur
  ON cur.ticker = p.ticker
 AND cur.month_end = p.month_end
 AND cur.period_rank = 1
 AND julianday(p.month_end) - julianday(cur.period_end) <= :max_staleness_days
LEFT JOIN periods AS prev
  ON prev.ticker = p.ticker
 AND prev.month_end = p.month_end
 AND prev.period_rank = 2
 AND cur.period_end IS NOT NULL
ORDER BY p.ticker, p.month_end;
