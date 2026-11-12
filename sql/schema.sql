-- Equity screener SQLite schema. Dates are ISO-8601 text (YYYY-MM-DD) so string
-- comparison and julianday() both work.

CREATE TABLE IF NOT EXISTS securities (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    sector        TEXT,
    sub_industry  TEXT,
    cik           TEXT,
    updated_at    TEXT
);

-- Month-end prices. `close` is split-adjusted only (used for valuation multiples);
-- `adj_close` is split- and dividend-adjusted (used for total returns).
CREATE TABLE IF NOT EXISTS prices_monthly (
    ticker     TEXT NOT NULL,
    month_end  TEXT NOT NULL,
    close      REAL,
    adj_close  REAL,
    PRIMARY KEY (ticker, month_end)
);

CREATE TABLE IF NOT EXISTS splits (
    ticker      TEXT NOT NULL,
    split_date  TEXT NOT NULL,
    ratio       REAL NOT NULL,          -- 4.0 for a 4-for-1 split
    PRIMARY KEY (ticker, split_date)
);

-- Annual fundamentals, normalised across providers and VERSIONED: one row per
-- (period, version). `available_date` is the first date that version could have been
-- known: the filing date when the provider gives one, otherwise period_end +
-- filing_lag_days (flagged by available_date_is_estimated), or the date a changed
-- value was first observed by this pipeline (version_type = 'revision_observed').
-- version_type: original | amendment (10-K/A) | restatement (changed comparative in a
-- later 10-K) | revision_observed (provider value changed between our downloads).
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker                      TEXT NOT NULL,
    period_end                  TEXT NOT NULL,
    source                      TEXT NOT NULL,
    available_date              TEXT NOT NULL,
    available_date_is_estimated INTEGER NOT NULL DEFAULT 0,
    version_type                TEXT NOT NULL DEFAULT 'original',
    accession                   TEXT,
    revenue                     REAL,
    ebit                        REAL,
    ebitda                      REAL,
    net_income                  REAL,
    eps_diluted                 REAL,
    shares_diluted              REAL,
    total_equity                REAL,
    total_debt                  REAL,
    cash                        REAL,
    operating_cash_flow         REAL,
    capex                       REAL,              -- capital expenditure as a positive amount spent
    total_assets                REAL,
    PRIMARY KEY (ticker, period_end, source, available_date)
);

-- S&P 500 additions and removals (Wikipedia "Selected changes" table) and the membership
-- spans reconstructed from them by walking back from today's constituents.
CREATE TABLE IF NOT EXISTS index_changes (
    change_date   TEXT NOT NULL,
    added         TEXT,
    added_name    TEXT,
    removed       TEXT,
    removed_name  TEXT
);

CREATE TABLE IF NOT EXISTS membership (
    ticker      TEXT NOT NULL,
    start_date  TEXT,                   -- NULL: member since before the earliest recorded change
    end_date    TEXT                    -- NULL: still a member
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    fetched_at_utc  TEXT NOT NULL,
    source          TEXT NOT NULL,
    endpoint        TEXT,
    ticker          TEXT,
    -- Per-ticker outcome: ok | cached | empty | error | rate_limited | skipped
    -- Per HTTP request (FMP, Alpha Vantage, SEC, Wikipedia): request_ok | request_retry | request_failed
    status          TEXT NOT NULL,
    n_rows          INTEGER,
    message         TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    started_at_utc   TEXT NOT NULL,
    finished_at_utc  TEXT,
    mode             TEXT,
    stages           TEXT,
    status           TEXT,
    message          TEXT
);

-- One row per stage execution. A run's status is derived from the latest attempt of each stage,
-- so a stage that failed and then succeeded on retry leaves the run successful, while the failed
-- attempt stays here as history.
CREATE TABLE IF NOT EXISTS pipeline_stage_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    stage            TEXT NOT NULL,
    attempt          INTEGER NOT NULL,
    started_at_utc   TEXT NOT NULL,
    finished_at_utc  TEXT,
    status           TEXT NOT NULL,      -- running | success | failed
    message          TEXT
);

CREATE TABLE IF NOT EXISTS quality_checks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    checked_at_utc  TEXT NOT NULL,
    check_name      TEXT NOT NULL,
    status          TEXT NOT NULL,     -- pass | warn | fail
    value           REAL,
    threshold       REAL,
    detail          TEXT
);

CREATE INDEX IF NOT EXISTS ix_prices_month ON prices_monthly (month_end);
CREATE INDEX IF NOT EXISTS ix_fund_avail ON fundamentals (ticker, source, available_date);
CREATE INDEX IF NOT EXISTS ix_fetch_run ON fetch_log (run_id);
