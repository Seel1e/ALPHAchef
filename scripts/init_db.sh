#!/usr/bin/env bash
# Creates application databases, users, schema, and indexes.
# Executed once by the Postgres entrypoint on first container start.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    -- ── Airflow metadata DB ──────────────────────────────────────────────
    CREATE DATABASE airflow;
    CREATE USER airflow WITH LOGIN PASSWORD 'airflow_pass';
    GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

    -- ── ALPHAchef market-data DB ─────────────────────────────────────────
    CREATE DATABASE alphachef;
    CREATE USER alphachef WITH LOGIN PASSWORD 'alphachef_pass';
    GRANT ALL PRIVILEGES ON DATABASE alphachef TO alphachef;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname alphachef <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- ── OHLCV price warehouse ─────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS historical_daily_prices (
        id          UUID            DEFAULT uuid_generate_v4() PRIMARY KEY,
        ticker      VARCHAR(10)     NOT NULL,
        trade_date  DATE            NOT NULL,
        open        NUMERIC(12, 4)  NOT NULL,
        high        NUMERIC(12, 4)  NOT NULL,
        low         NUMERIC(12, 4)  NOT NULL,
        close       NUMERIC(12, 4)  NOT NULL,
        adj_close   NUMERIC(12, 4)  NOT NULL,
        volume      BIGINT          NOT NULL
    );

    -- Fast composite lookup (required by ETL upsert)
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker_date
        ON historical_daily_prices (ticker, trade_date);

    -- BRIN index for efficient date-range scans
    CREATE INDEX IF NOT EXISTS idx_trade_date_brin
        ON historical_daily_prices USING BRIN (trade_date);

    -- ── EGARCH residuals ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS egarch_residuals (
        id           UUID            DEFAULT uuid_generate_v4() PRIMARY KEY,
        ticker       VARCHAR(10)     NOT NULL,
        trade_date   DATE            NOT NULL,
        log_return   NUMERIC(16, 10),
        std_residual NUMERIC(16, 10),
        cond_vol     NUMERIC(16, 10),
        run_ts       TIMESTAMPTZ     DEFAULT NOW()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_residuals_ticker_date
        ON egarch_residuals (ticker, trade_date);

    -- ── Monte-Carlo simulation results ────────────────────────────────────
    CREATE TABLE IF NOT EXISTS simulation_results (
        id             UUID            DEFAULT uuid_generate_v4() PRIMARY KEY,
        run_id         UUID            NOT NULL,
        run_ts         TIMESTAMPTZ     DEFAULT NOW(),
        horizon_days   INT             NOT NULL,
        n_simulations  INT             NOT NULL,
        var_99         NUMERIC(12, 6),
        cvar_99        NUMERIC(12, 6),
        median_return  NUMERIC(12, 6),
        mean_return    NUMERIC(12, 6),
        std_return     NUMERIC(12, 6),
        skewness       NUMERIC(12, 6),
        kurtosis       NUMERIC(12, 6),
        pct_loss_gt10  NUMERIC(8, 6),
        pct_loss_gt20  NUMERIC(8, 6),
        copula_dof     NUMERIC(10, 4),
        tickers        TEXT[]
    );

    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alphachef;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alphachef;
EOSQL
