"""
ETL extraction: pull OHLCV data from yfinance and upsert into PostgreSQL.

Idempotent by design — re-running never creates duplicates.
Incremental by default — only fetches missing dates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB, PORTFOLIO

logger = logging.getLogger(__name__)


# ── Engine factory ─────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    return create_engine(DB.url, pool_pre_ping=True, pool_size=5, max_overflow=10)


# ── Download ───────────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download adjusted OHLCV for one ticker via yfinance."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )

    if raw.empty:
        logger.warning("yfinance returned no data for %s [%s → %s]", ticker, start, end)
        return pd.DataFrame()

    # yfinance ≥ 0.2.18 returns a MultiIndex when auto_adjust=False
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [lvl0 for lvl0, _ in raw.columns]

    raw = raw.reset_index()
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    rename_map = {"date": "trade_date", "adj_close": "adj_close"}
    raw = raw.rename(columns=rename_map)
    raw["ticker"] = ticker

    keep = ["ticker", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]
    raw = raw[[c for c in keep if c in raw.columns]].dropna(subset=["close"])
    raw["volume"] = raw["volume"].fillna(0).astype("int64")
    return raw


# ── Upsert ─────────────────────────────────────────────────────────────────────

_UPSERT_SQL = text("""
    INSERT INTO historical_daily_prices
        (ticker, trade_date, open, high, low, close, adj_close, volume)
    VALUES
        (:ticker, :trade_date, :open, :high, :low, :close, :adj_close, :volume)
    ON CONFLICT (ticker, trade_date) DO UPDATE SET
        open      = EXCLUDED.open,
        high      = EXCLUDED.high,
        low       = EXCLUDED.low,
        close     = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume    = EXCLUDED.volume
""")


def upsert_prices(df: pd.DataFrame, engine: Engine) -> int:
    if df.empty:
        return 0
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, df.to_dict("records"))
    return len(df)


# ── Incremental helper ─────────────────────────────────────────────────────────

def _last_date(ticker: str, engine: Engine, fallback: str) -> str:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(trade_date) FROM historical_daily_prices WHERE ticker = :t"),
            {"t": ticker},
        ).scalar()
    if result is not None:
        return (result + timedelta(days=1)).strftime("%Y-%m-%d")
    return fallback


# ── Main entry point ───────────────────────────────────────────────────────────

def run_etl(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    incremental: bool = True,
) -> Dict[str, dict]:
    """
    Pull data for all tickers and upsert to PostgreSQL.

    Returns per-ticker status dict.
    """
    tickers = tickers or PORTFOLIO.tickers
    start = start or PORTFOLIO.start_date
    engine = get_engine()
    summary: Dict[str, dict] = {}

    for ticker in tickers:
        try:
            effective_start = _last_date(ticker, engine, start) if incremental else start
            df = fetch_ticker(ticker, effective_start, end)
            n = upsert_prices(df, engine)
            summary[ticker] = {"rows": n, "status": "ok"}
            logger.info("%-6s  upserted %d rows", ticker, n)
        except Exception as exc:
            logger.exception("ETL failed for %s", ticker)
            summary[ticker] = {"rows": 0, "status": "error", "error": str(exc)}

    return summary
