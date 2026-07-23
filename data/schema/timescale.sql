-- §2.3 — TimescaleDB schema
-- Run: psql $DB_URL -f data/schema/timescale.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Feature table: one row per asset per minute
CREATE TABLE IF NOT EXISTS market_features (
    ts              TIMESTAMPTZ     NOT NULL,
    ticker          TEXT            NOT NULL,
    close           FLOAT8,
    volume          FLOAT8,
    bid             FLOAT8,
    ask             FLOAT8,
    vwap            FLOAT8,
    leverage        FLOAT8  DEFAULT 1.0,        -- ℓ_t (capital structure)
    shares_outstanding FLOAT8,                  -- κ_t
    disclosure_level   FLOAT8  DEFAULT 0.5,     -- ι_t ∈ [0,1]
    rsi_14          FLOAT8,
    bpv_1d          FLOAT8,                     -- bipower variation → σ_τ² estimate
    jump_flag       BOOLEAN DEFAULT FALSE,       -- behavioral noise spike (η)
    news_emb        FLOAT8[],                   -- 384-dim sentence-transformer embedding
    macro_state     FLOAT8[]                    -- [DFF, CPIAUCSL, UNRATE, T10Y2Y, VIX, M2SL]
);

SELECT create_hypertable('market_features', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_features_ticker_ts
    ON market_features (ticker, ts DESC);

-- Continuous aggregate: daily summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_daily
WITH (timescaledb.continuous) AS
SELECT
    ticker,
    time_bucket('1 day', ts)   AS day,
    LAST(close, ts)             AS close,
    SUM(volume)                 AS volume,
    AVG(bpv_1d)                 AS avg_bpv,
    BOOL_OR(jump_flag)          AS had_jump
FROM market_features
GROUP BY ticker, day
WITH NO DATA;

-- Dual noise calibration results (one row per ticker per day)
CREATE TABLE IF NOT EXISTS noise_params (
    day             DATE        NOT NULL,
    ticker          TEXT        NOT NULL,
    sigma_tau       FLOAT8,     -- physical noise volatility
    lambda_eta      FLOAT8,     -- behavioral jump intensity
    m2_eta          FLOAT8,     -- second moment of jump sizes
    tau_composite   FLOAT8,     -- τ_t = sqrt(σ_τ² + λ_η · m₂^η)
    PRIMARY KEY (day, ticker)
);

-- MFG equilibrium snapshots
CREATE TABLE IF NOT EXISTS mfg_equilibrium (
    ts              TIMESTAMPTZ NOT NULL,
    latent_state    FLOAT8[],   -- z_t ∈ ℝ^64
    mu_star         FLOAT8[],   -- equilibrium drift m*(z_t)
    mfg_residual    FLOAT8,     -- W₂ distance at convergence
    regime          TEXT,       -- 'normal' | 'crisis' | 'recovery'
    lyapunov_index  FLOAT8      -- LV(z)/V(z) instability indicator
);

SELECT create_hypertable('mfg_equilibrium', 'ts', if_not_exists => TRUE);
