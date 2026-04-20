"""
Creates the ALPHAchef PostgreSQL database, user, tables, and indexes.
Run once before the pipeline — safe to re-run (idempotent).

Usage:
    python scripts/setup_db.py
"""

import os
import sys

# Must connect as a superuser to CREATE DATABASE / ROLE
PG_SUPERUSER = os.getenv("PG_SUPERUSER", "postgres")
PG_SUPERPASS = os.getenv("PG_SUPERPASS", "postgres_master")
PG_HOST      = os.getenv("PG_HOST", "localhost")
PG_PORT      = os.getenv("PG_PORT", "5432")


def run():
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        sys.exit("psycopg2 not found — activate your venv and run: pip install psycopg2-binary")

    # ── Connect as superuser to postgres DB ───────────────────────────────
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname="postgres",
        user=PG_SUPERUSER, password=PG_SUPERPASS,
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # ── Create role & database (skip if already exist) ────────────────────
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'alphachef'")
    if not cur.fetchone():
        cur.execute("CREATE USER alphachef WITH LOGIN PASSWORD 'alphachef_pass'")
        print("Created user: alphachef")

    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'alphachef'")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE alphachef OWNER alphachef")
        print("Created database: alphachef")
    else:
        print("Database 'alphachef' already exists — skipping creation.")

    cur.close()
    conn.close()

    # ── Connect to alphachef DB and create schema ─────────────────────────
    conn2 = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname="alphachef",
        user=PG_SUPERUSER, password=PG_SUPERPASS,
    )
    conn2.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur2 = conn2.cursor()

    ddl = """
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker_date
        ON historical_daily_prices (ticker, trade_date);
    CREATE INDEX IF NOT EXISTS idx_trade_date_brin
        ON historical_daily_prices USING BRIN (trade_date);

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

    GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA public TO alphachef;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alphachef;
    """

    cur2.execute(ddl)
    print("Schema, tables, and indexes created (or already exist).")
    cur2.close()
    conn2.close()
    print("\nSetup complete. You can now run:  python run_pipeline.py --phase all --no-spark")


if __name__ == "__main__":
    run()
