"""Shared database helpers for the Streamlit dashboard."""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DB


@st.cache_resource
def get_engine() -> Engine:
    return create_engine(DB.url, pool_pre_ping=True, pool_size=3)


@st.cache_data(ttl=300)
def load_prices(tickers: list | None = None) -> pd.DataFrame:
    engine = get_engine()
    where = "WHERE ticker = ANY(:t)" if tickers else ""
    params = {"t": tickers} if tickers else {}
    with engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM historical_daily_prices {where} ORDER BY ticker, trade_date"),
            conn,
            params=params,
            parse_dates=["trade_date"],
        )


@st.cache_data(ttl=300)
def load_egarch_residuals(tickers: list | None = None) -> pd.DataFrame:
    engine = get_engine()
    where = "WHERE ticker = ANY(:t)" if tickers else ""
    params = {"t": tickers} if tickers else {}
    with engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM egarch_residuals {where} ORDER BY ticker, trade_date"),
            conn,
            params=params,
            parse_dates=["trade_date"],
        )


@st.cache_data(ttl=60)
def load_latest_simulation() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT * FROM simulation_results ORDER BY run_ts DESC LIMIT 20"),
            conn,
            parse_dates=["run_ts"],
        )


def has_data(table: str = "historical_daily_prices") -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        return n > 0
    except Exception:
        return False
