-- ============================================================
-- Retail Demand Forecasting Platform
-- PostgreSQL Schema v1.0
-- Authored for scientific rigor -- all tables normalized to 3NF
-- ============================================================

-- ── Dimension: Items ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_item (
    item_id      VARCHAR(50)  PRIMARY KEY,
    dept_id      VARCHAR(30)  NOT NULL,
    cat_id       VARCHAR(30)  NOT NULL,
    store_id     VARCHAR(20)  NOT NULL,
    state_id     VARCHAR(10)  NOT NULL,
    created_at   TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_item_dept   ON dim_item(dept_id);
CREATE INDEX IF NOT EXISTS idx_dim_item_cat    ON dim_item(cat_id);
CREATE INDEX IF NOT EXISTS idx_dim_item_store  ON dim_item(store_id);

-- ── Dimension: Calendar ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_calendar (
    date_id          DATE         PRIMARY KEY,
    wm_yr_wk         INTEGER,
    weekday          VARCHAR(10),
    wday             SMALLINT,    -- 1=Saturday ... 7=Friday
    month            SMALLINT,
    year             SMALLINT,
    event_name_1     VARCHAR(100),
    event_type_1     VARCHAR(50),
    event_name_2     VARCHAR(100),
    event_type_2     VARCHAR(50),
    snap_ca          BOOLEAN      DEFAULT FALSE,
    snap_tx          BOOLEAN      DEFAULT FALSE,
    snap_wi          BOOLEAN      DEFAULT FALSE,
    is_holiday       BOOLEAN GENERATED ALWAYS AS
        (event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL) STORED,
    created_at       TIMESTAMP    DEFAULT NOW()
);

-- ── Dimension: Stores ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dim_store (
    store_id    VARCHAR(20)  PRIMARY KEY,
    state_id    VARCHAR(10)  NOT NULL,
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- ── Fact: Daily Sales ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_sales (
    id           BIGSERIAL    PRIMARY KEY,
    item_id      VARCHAR(50)  NOT NULL REFERENCES dim_item(item_id),
    store_id     VARCHAR(20)  NOT NULL REFERENCES dim_store(store_id),
    date_id      DATE         NOT NULL REFERENCES dim_calendar(date_id),
    sales        FLOAT        NOT NULL CHECK (sales >= 0),
    loaded_at    TIMESTAMP    DEFAULT NOW(),
    UNIQUE (item_id, store_id, date_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_item_store_date
    ON fact_sales(item_id, store_id, date_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_date
    ON fact_sales(date_id);

-- ── Fact: Sell Prices ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_sell_prices (
    id           BIGSERIAL    PRIMARY KEY,
    store_id     VARCHAR(20)  NOT NULL REFERENCES dim_store(store_id),
    item_id      VARCHAR(50)  NOT NULL REFERENCES dim_item(item_id),
    wm_yr_wk     INTEGER      NOT NULL,
    sell_price   FLOAT        NOT NULL CHECK (sell_price > 0),
    loaded_at    TIMESTAMP    DEFAULT NOW(),
    UNIQUE (store_id, item_id, wm_yr_wk)
);

CREATE INDEX IF NOT EXISTS idx_sell_prices_item_store
    ON fact_sell_prices(item_id, store_id);

-- ── Fact: Forecasts (model outputs) ──────────────────────────
CREATE TABLE IF NOT EXISTS fact_forecasts (
    id              BIGSERIAL    PRIMARY KEY,
    item_id         VARCHAR(50)  NOT NULL,
    store_id        VARCHAR(20)  NOT NULL,
    date_id         DATE         NOT NULL,
    model_name      VARCHAR(50)  NOT NULL,
    forecast        FLOAT        NOT NULL,
    lower_ci_95     FLOAT,
    upper_ci_95     FLOAT,
    run_timestamp   TIMESTAMP    DEFAULT NOW(),
    UNIQUE (item_id, store_id, date_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_item_store_date
    ON fact_forecasts(item_id, store_id, date_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_model
    ON fact_forecasts(model_name);

-- ── Fact: Risk Flags ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_risk_flags (
    id              BIGSERIAL    PRIMARY KEY,
    item_id         VARCHAR(50)  NOT NULL,
    store_id        VARCHAR(20)  NOT NULL,
    date_id         DATE         NOT NULL,
    flag_type       VARCHAR(50)  NOT NULL,  -- 'isolation_forest','shortfall','high_volatility'
    anomaly_score   FLOAT,
    threshold       FLOAT,
    is_anomaly      BOOLEAN      NOT NULL DEFAULT FALSE,
    rolling_cv      FLOAT,
    rolling_mean    FLOAT,
    rolling_std     FLOAT,
    run_timestamp   TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_flags_item_store_date
    ON fact_risk_flags(item_id, store_id, date_id);
CREATE INDEX IF NOT EXISTS idx_risk_flags_type
    ON fact_risk_flags(flag_type);

-- ── Metrics: Model Evaluation Results ────────────────────────
CREATE TABLE IF NOT EXISTS model_evaluation_results (
    id              BIGSERIAL    PRIMARY KEY,
    model_name      VARCHAR(50)  NOT NULL,
    item_id         VARCHAR(50),           -- NULL = aggregate across all
    store_id        VARCHAR(20),
    fold_index      SMALLINT,
    horizon_days    SMALLINT,
    mape            FLOAT,
    rmse            FLOAT,
    mae             FLOAT,
    smape           FLOAT,
    wrmsse          FLOAT,
    coverage_95     FLOAT,                 -- fraction of actuals inside 95% CI
    interval_width  FLOAT,                 -- avg width of 95% CI
    train_start     DATE,
    train_end       DATE,
    test_start      DATE,
    test_end        DATE,
    run_timestamp   TIMESTAMP    DEFAULT NOW()
);

-- ── Metrics: Pipeline Run Log ─────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    id              BIGSERIAL    PRIMARY KEY,
    dag_run_id      VARCHAR(200),
    stage           VARCHAR(100) NOT NULL,
    status          VARCHAR(20)  NOT NULL,  -- 'success','failed','skipped'
    rows_processed  BIGINT,
    duration_secs   FLOAT,
    error_message   TEXT,
    run_timestamp   TIMESTAMP    DEFAULT NOW()
);