-- ============================================================
-- FOREX AI TRADING AGENT — Database Schema v1
-- PostgreSQL 16 + pgvector + TimescaleDB
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector for RAG embeddings
CREATE EXTENSION IF NOT EXISTS timescaledb;     -- time-series hypertables
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- trigram index for text search

-- ============================================================
-- 1. CANDLES — Raw price data (TimescaleDB hypertable)
-- ============================================================
CREATE TABLE candles (
    instrument   VARCHAR(10)    NOT NULL,
    timeframe    VARCHAR(5)     NOT NULL,   -- '1m','5m','15m','1h','4h','1d'
    ts           TIMESTAMPTZ    NOT NULL,
    open         DECIMAL(12,6)  NOT NULL,
    high         DECIMAL(12,6)  NOT NULL,
    low          DECIMAL(12,6)  NOT NULL,
    close        DECIMAL(12,6)  NOT NULL,
    volume       DECIMAL(18,2)  NOT NULL DEFAULT 0,
    PRIMARY KEY (instrument, timeframe, ts)
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
-- Partitions by 1 week, which balances insert speed and query performance
-- for 9 instruments × 6 timeframes at high frequency
SELECT create_hypertable(
    'candles',
    by_range('ts', INTERVAL '1 week'),
    migrate_data => true
);

-- Composite indexes for the most common query patterns
CREATE INDEX idx_candles_instrument_tf_ts
    ON candles (instrument, timeframe, ts DESC);

-- ============================================================
-- 2. MARKET EVENTS — Historical Brain knowledge base (RAG)
-- ============================================================
CREATE TABLE market_events (
    id              SERIAL          PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    event_type      VARCHAR(50)     NOT NULL,   -- 'central_bank','economic_release','geopolitical','technical_pattern','regime_change'
    event_name      VARCHAR(200)    NOT NULL,
    description     TEXT,
    affected_pairs  TEXT[]          NOT NULL DEFAULT '{}',
    market_context  JSONB           NOT NULL DEFAULT '{}',  -- macro regime, correlations at time
    price_impact    JSONB           NOT NULL DEFAULT '{}',  -- measured moves per instrument
    tags            TEXT[]          NOT NULL DEFAULT '{}',
    embedding       vector(1536),   -- Claude/OpenAI embedding for similarity search
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor on embeddings
-- ef_construction=128 gives good recall; m=16 is standard for 1536-dim
CREATE INDEX idx_market_events_embedding
    ON market_events
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX idx_market_events_type     ON market_events (event_type);
CREATE INDEX idx_market_events_ts       ON market_events (ts DESC);
CREATE INDEX idx_market_events_pairs    ON market_events USING GIN (affected_pairs);
CREATE INDEX idx_market_events_tags     ON market_events USING GIN (tags);

-- ============================================================
-- 3. TRADES — Full audit trail
-- ============================================================
CREATE TABLE trades (
    id              SERIAL          PRIMARY KEY,
    trade_id        VARCHAR(50)     UNIQUE NOT NULL,  -- 'T-YYYYMMDD-NNN'
    ts              TIMESTAMPTZ     NOT NULL,
    instrument      VARCHAR(10)     NOT NULL,
    direction       VARCHAR(5)      NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    entry_price     DECIMAL(12,6)   NOT NULL,
    stop_loss       DECIMAL(12,6)   NOT NULL,
    take_profit_1   DECIMAL(12,6),
    take_profit_2   DECIMAL(12,6),
    position_size   DECIMAL(8,4)    NOT NULL,  -- % of capital
    confidence      DECIMAL(4,2)    NOT NULL,
    status          VARCHAR(20)     NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','open','closed_win','closed_loss','cancelled','vetoed')),
    exit_price      DECIMAL(12,6),
    exit_ts         TIMESTAMPTZ,
    pnl_pips        DECIMAL(8,2),
    pnl_percent     DECIMAL(6,3),
    agent_reasoning JSONB           NOT NULL DEFAULT '{}',  -- full agent votes and reasoning
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_instrument  ON trades (instrument, ts DESC);
CREATE INDEX idx_trades_status      ON trades (status);
CREATE INDEX idx_trades_date        ON trades (ts DESC);

-- ============================================================
-- 4. AGENT DECISIONS — Every agent output logged
-- ============================================================
CREATE TABLE agent_decisions (
    id              SERIAL          PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    agent_name      VARCHAR(50)     NOT NULL,
    instrument      VARCHAR(10),                      -- NULL for cross-instrument agents
    decision        JSONB           NOT NULL,          -- full agent output
    confidence      DECIMAL(4,2),
    execution_ms    INTEGER,                          -- how long the agent took
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Hypertable for agent decisions — they accumulate fast
SELECT create_hypertable(
    'agent_decisions',
    by_range('ts', INTERVAL '1 day'),
    migrate_data => true
);

CREATE INDEX idx_agent_decisions_agent  ON agent_decisions (agent_name, ts DESC);

-- ============================================================
-- 5. CORRELATION MATRIX — Periodic snapshots
-- ============================================================
CREATE TABLE correlation_matrix (
    id              SERIAL          PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    window_days     INTEGER         NOT NULL,   -- 30, 90, 252
    matrix          JSONB           NOT NULL,   -- full NxN correlation matrix
    anomalies       JSONB           NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_corr_ts ON correlation_matrix (ts DESC);

-- ============================================================
-- 6. DAILY PERFORMANCE — Aggregated daily metrics
-- ============================================================
CREATE TABLE daily_performance (
    id              SERIAL          PRIMARY KEY,
    date            DATE            UNIQUE NOT NULL,
    total_trades    INTEGER         NOT NULL DEFAULT 0,
    wins            INTEGER         NOT NULL DEFAULT 0,
    losses          INTEGER         NOT NULL DEFAULT 0,
    win_rate        DECIMAL(5,2),
    total_pnl_pips  DECIMAL(10,2)   NOT NULL DEFAULT 0,
    total_pnl_pct   DECIMAL(6,3)    NOT NULL DEFAULT 0,
    max_drawdown    DECIMAL(6,3),
    profit_factor   DECIMAL(6,2),
    sharpe_ratio    DECIMAL(6,2),
    best_trade      JSONB,
    worst_trade     JSONB,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 7. ECONOMIC CALENDAR — Scheduled events
-- ============================================================
CREATE TABLE economic_events (
    id              SERIAL          PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    country         VARCHAR(5)      NOT NULL,   -- 'US','EU','GB','JP','CH','AU','CA','NZ'
    event_name      VARCHAR(200)    NOT NULL,
    impact          VARCHAR(10)     NOT NULL CHECK (impact IN ('HIGH','MEDIUM','LOW')),
    forecast        VARCHAR(50),
    previous        VARCHAR(50),
    actual          VARCHAR(50),
    currency        VARCHAR(3),
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(ts, country, event_name)
);

CREATE INDEX idx_econ_events_ts     ON economic_events (ts);
CREATE INDEX idx_econ_events_impact ON economic_events (impact, ts);

-- ============================================================
-- 8. NEWS HEADLINES — For sentiment analysis
-- ============================================================
CREATE TABLE news_headlines (
    id              SERIAL          PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL,
    source          VARCHAR(100)    NOT NULL,
    headline        TEXT            NOT NULL,
    url             TEXT,
    currencies      TEXT[]          NOT NULL DEFAULT '{}',
    sentiment       VARCHAR(10),    -- 'bullish','bearish','neutral' (set by Sentiment Agent)
    sentiment_score DECIMAL(4,2),
    processed       BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'news_headlines',
    by_range('ts', INTERVAL '1 day'),
    migrate_data => true
);

CREATE INDEX idx_news_processed ON news_headlines (processed, ts DESC);

-- ============================================================
-- 9. RISK STATE — Current risk manager state (single row, upserted)
-- ============================================================
CREATE TABLE risk_state (
    id                      INTEGER         PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- singleton
    daily_pnl_pct           DECIMAL(6,3)    NOT NULL DEFAULT 0,
    weekly_pnl_pct          DECIMAL(6,3)    NOT NULL DEFAULT 0,
    consecutive_losses      INTEGER         NOT NULL DEFAULT 0,
    open_trade_count        INTEGER         NOT NULL DEFAULT 0,
    open_instruments        TEXT[]          NOT NULL DEFAULT '{}',
    circuit_breaker_until   TIMESTAMPTZ,
    daily_halt              BOOLEAN         NOT NULL DEFAULT FALSE,
    weekly_halt             BOOLEAN         NOT NULL DEFAULT FALSE,
    last_updated            TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Initialize singleton
INSERT INTO risk_state (id) VALUES (1);

-- ============================================================
-- Trigger: auto-update updated_at on trades
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
