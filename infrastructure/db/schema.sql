-- =============================================================================
-- TimescaleDB Schema for Systematic Volatility Portfolio
-- =============================================================================
-- Run against a PostgreSQL database with TimescaleDB extension enabled.
-- docker exec -i vol-timescaledb psql -U vol_user -d vol_db < schema.sql
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- 1. TICK DATA — raw IBKR market data ingestion
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS equity_ticks (
    ts              TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    price           DOUBLE PRECISION,
    size            BIGINT,
    exchange        TEXT,
    tick_type       TEXT            -- LAST, BID, ASK, OPEN, HIGH, LOW, CLOSE
);
SELECT create_hypertable('equity_ticks', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS option_ticks (
    ts              TIMESTAMPTZ     NOT NULL,
    underlying      TEXT            NOT NULL,
    expiry          DATE            NOT NULL,
    strike          DOUBLE PRECISION NOT NULL,
    right           TEXT            NOT NULL,   -- C or P
    bid             DOUBLE PRECISION,
    ask             DOUBLE PRECISION,
    last            DOUBLE PRECISION,
    volume          BIGINT,
    open_interest   BIGINT,
    implied_vol     DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    gamma           DOUBLE PRECISION,
    vega            DOUBLE PRECISION,
    theta           DOUBLE PRECISION
);
SELECT create_hypertable('option_ticks', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 2. IMPLIED VOL SURFACE — matches strategies/dispersion.py expectations
--    Columns: name, strike_delta, tenor_months, implied_vol,
--             forward_price, spot_price, index_weight, sector
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS implied_vol_surface (
    ts              TIMESTAMPTZ     NOT NULL,
    name            TEXT            NOT NULL,
    strike_delta    TEXT            NOT NULL,   -- 10D_PUT, 25D_PUT, ATM, 25D_CALL, 10D_CALL
    tenor_months    INTEGER         NOT NULL,   -- 1, 2, 3, 6, 9, 12
    implied_vol     DOUBLE PRECISION NOT NULL,
    forward_price   DOUBLE PRECISION,
    spot_price      DOUBLE PRECISION,
    index_weight    DOUBLE PRECISION,
    sector          TEXT
);
SELECT create_hypertable('implied_vol_surface', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 3. REALIZED VOL HISTORY — matches realized_vol_history.csv
--    One row per asset per day with return and rolling realized vol
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS realized_vol_history (
    ts              TIMESTAMPTZ     NOT NULL,
    name            TEXT            NOT NULL,
    daily_return    DOUBLE PRECISION,
    realized_vol    DOUBLE PRECISION    -- 20-day rolling annualized
);
SELECT create_hypertable('realized_vol_history', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 4. CORRELATION MATRIX — 90-day rolling pairwise correlations
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS correlation_snapshots (
    ts              TIMESTAMPTZ     NOT NULL,
    asset_1         TEXT            NOT NULL,
    asset_2         TEXT            NOT NULL,
    correlation     DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('correlation_snapshots', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 5. VOL REGIME SIGNALS — matches vol_regime_signals.csv
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS vol_regime_signals (
    ts                      TIMESTAMPTZ     NOT NULL,
    vix_front_month         DOUBLE PRECISION,
    vix_second_month        DOUBLE PRECISION,
    term_structure_slope    DOUBLE PRECISION,
    vvix                    DOUBLE PRECISION,
    realized_vol_20d        DOUBLE PRECISION,
    rv_iv_spread            DOUBLE PRECISION,
    regime                  TEXT            -- LOW_VOL_HARVESTING, TRANSITIONAL, CRISIS
);
SELECT create_hypertable('vol_regime_signals', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 6. STRATEGY SIGNALS — output of each strategy's generate_signals()
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS strategy_signals (
    ts              TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL,
    signal_name     TEXT            NOT NULL,
    signal_value    DOUBLE PRECISION,
    metadata        JSONB
);
SELECT create_hypertable('strategy_signals', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 7. POSITIONS — current and historical strategy positions
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS positions (
    ts              TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL,
    leg             TEXT            NOT NULL,
    instrument      TEXT            NOT NULL,
    direction       TEXT            NOT NULL,   -- LONG / SHORT
    notional_usd    DOUBLE PRECISION,
    delta_usd       DOUBLE PRECISION,
    gamma_usd       DOUBLE PRECISION,
    vega_usd        DOUBLE PRECISION,
    theta_usd       DOUBLE PRECISION,
    regime          TEXT,
    metadata        JSONB
);
SELECT create_hypertable('positions', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 8. PORTFOLIO GREEKS — aggregated daily snapshot
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS portfolio_greeks (
    ts                      TIMESTAMPTZ     NOT NULL,
    strategy                TEXT            NOT NULL,   -- per-strategy + 'PORTFOLIO_TOTAL'
    capital_allocated       DOUBLE PRECISION,
    notional_deployed       DOUBLE PRECISION,
    delta_usd               DOUBLE PRECISION,
    gamma_usd               DOUBLE PRECISION,
    vega_usd                DOUBLE PRECISION,
    theta_usd               DOUBLE PRECISION,
    vega_headroom_floor     DOUBLE PRECISION,
    vega_headroom_ceiling   DOUBLE PRECISION
);
SELECT create_hypertable('portfolio_greeks', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 9. SCENARIO RESULTS — stress test outputs
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS scenario_results (
    ts                  TIMESTAMPTZ     NOT NULL,
    scenario_name       TEXT            NOT NULL,
    vix_level           DOUBLE PRECISION,
    correlation_shock   DOUBLE PRECISION,
    strategy            TEXT            NOT NULL,
    pnl_usd             DOUBLE PRECISION,
    pnl_pct_capital     DOUBLE PRECISION
);
SELECT create_hypertable('scenario_results', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 10. PNL ATTRIBUTION — daily decomposition
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pnl_attribution (
    ts              TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL,
    vega_pnl        DOUBLE PRECISION,
    gamma_pnl       DOUBLE PRECISION,
    theta_pnl       DOUBLE PRECISION,
    correlation_pnl DOUBLE PRECISION,
    residual_pnl    DOUBLE PRECISION,
    total_pnl       DOUBLE PRECISION
);
SELECT create_hypertable('pnl_attribution', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- 11. EXECUTION LOG — trade fills and costs
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS execution_log (
    ts              TIMESTAMPTZ     NOT NULL,
    strategy        TEXT            NOT NULL,
    order_id        TEXT,
    instrument      TEXT            NOT NULL,
    direction       TEXT            NOT NULL,
    quantity        DOUBLE PRECISION,
    fill_price      DOUBLE PRECISION,
    commission      DOUBLE PRECISION,
    slippage_bps    DOUBLE PRECISION,
    phase           TEXT,           -- PHASE_1, PHASE_2, PHASE_3
    metadata        JSONB
);
SELECT create_hypertable('execution_log', 'ts', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- INDEXES for common query patterns
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_equity_ticks_symbol ON equity_ticks (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_option_ticks_underlying ON option_ticks (underlying, ts DESC);
CREATE INDEX IF NOT EXISTS idx_option_ticks_chain ON option_ticks (underlying, expiry, strike, right, ts DESC);
CREATE INDEX IF NOT EXISTS idx_iv_surface_name ON implied_vol_surface (name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rv_history_name ON realized_vol_history (name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_regime_ts ON vol_regime_signals (ts DESC);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions (strategy, ts DESC);
CREATE INDEX IF NOT EXISTS idx_pnl_strategy ON pnl_attribution (strategy, ts DESC);
CREATE INDEX IF NOT EXISTS idx_greeks_strategy ON portfolio_greeks (strategy, ts DESC);

-- ---------------------------------------------------------------------------
-- COMPRESSION POLICIES — compress tick data older than 7 days
-- ---------------------------------------------------------------------------

ALTER TABLE equity_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('equity_ticks', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE option_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'underlying',
    timescaledb.compress_orderby = 'ts DESC'
);
SELECT add_compression_policy('option_ticks', INTERVAL '7 days', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- RETENTION POLICIES — drop raw ticks older than 90 days (keep aggregates)
-- ---------------------------------------------------------------------------

SELECT add_retention_policy('equity_ticks', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('option_ticks', INTERVAL '90 days', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- CONTINUOUS AGGREGATES — pre-computed OHLCV bars
-- ---------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS equity_bars_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts)     AS bucket,
    symbol,
    first(price, ts)                AS open,
    max(price)                      AS high,
    min(price)                      AS low,
    last(price, ts)                 AS close,
    sum(size)                       AS volume
FROM equity_ticks
WHERE tick_type = 'LAST'
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('equity_bars_1m',
    start_offset    => INTERVAL '1 hour',
    end_offset      => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists   => TRUE
);
