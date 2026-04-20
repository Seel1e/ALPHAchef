"""
Phase 2 — Time-Series Econometrics

Pipeline:
  1. Load adj_close from PostgreSQL → compute log-returns
  2. Augmented Dickey-Fuller (ADF) test to confirm stationarity
  3. Fit EGARCH(1,1) with Student-t errors (leverage-effect aware)
  4. Persist standardised residuals + conditional volatility to egarch_residuals
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from sqlalchemy import text
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import adfuller

from src.config import MODEL, PORTFOLIO
from src.etl.extractor import get_engine

logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_log_returns(
    tickers: List[str],
    engine: Engine,
    start: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query adj_close from the warehouse, pivot to wide format, return log-returns.
    Returns a DataFrame with tickers as columns and trade_date as index.
    """
    where_start = "AND trade_date >= :start" if start else ""
    query = text(f"""
        SELECT ticker, trade_date, adj_close
        FROM historical_daily_prices
        WHERE ticker = ANY(:tickers)
        {where_start}
        ORDER BY ticker, trade_date
    """)
    params: dict = {"tickers": tickers}
    if start:
        params["start"] = start

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["trade_date"])

    if df.empty:
        raise ValueError("No price data found. Run the ETL phase first.")

    wide = df.pivot(index="trade_date", columns="ticker", values="adj_close").sort_index()
    log_ret = np.log(wide / wide.shift(1)).dropna()
    logger.info("Log-returns: %d observations × %d assets", *log_ret.shape)
    return log_ret


# ── ADF stationarity test ──────────────────────────────────────────────────────

def adf_test(series: pd.Series, ticker: str) -> dict:
    stat, pval, _, _, crit, _ = adfuller(series.dropna(), autolag="AIC")
    stationary = pval < 0.05
    logger.info(
        "ADF  %-6s  stat=%.4f  p=%.4f  stationary=%s",
        ticker, stat, pval, stationary,
    )
    return {
        "ticker": ticker,
        "adf_statistic": stat,
        "p_value": pval,
        "is_stationary": stationary,
        "critical_1pct": crit["1%"],
        "critical_5pct": crit["5%"],
    }


# ── EGARCH(1,1) fitting ────────────────────────────────────────────────────────

def fit_egarch(
    series: pd.Series,
    ticker: str,
) -> Tuple[object, pd.Series, pd.Series]:
    """
    Fit EGARCH(1,1) with Student-t innovations.

    Captures the leverage effect: negative shocks increase volatility
    more than positive shocks of equal magnitude.

    Returns
    -------
    res          : arch ModelResult
    std_resids   : standardised residuals  ε_t / σ_t
    cond_vol     : daily conditional volatility σ_t  (decimal, not %)
    """
    scaled = series * 100  # percent scale for numerical stability

    am = arch_model(
        scaled,
        mean="AR",
        lags=1,
        vol="EGARCH",
        p=MODEL.egarch_p,
        q=MODEL.egarch_q,
        dist="t",
    )
    res = am.fit(disp="off", options={"maxiter": 600, "ftol": 1e-9})

    std_resids = res.std_resid.dropna()
    cond_vol = res.conditional_volatility.dropna() / 100  # → decimal

    logger.info(
        "EGARCH %-6s  AIC=%.1f  BIC=%.1f  LL=%.1f",
        ticker, res.aic, res.bic, res.loglikelihood,
    )
    return res, std_resids, cond_vol


# ── Persistence ────────────────────────────────────────────────────────────────

_UPSERT_SQL = text("""
    INSERT INTO egarch_residuals
        (ticker, trade_date, log_return, std_residual, cond_vol)
    VALUES
        (:ticker, :trade_date, :log_return, :std_residual, :cond_vol)
    ON CONFLICT (ticker, trade_date) DO UPDATE SET
        log_return   = EXCLUDED.log_return,
        std_residual = EXCLUDED.std_residual,
        cond_vol     = EXCLUDED.cond_vol,
        run_ts       = NOW()
""")


def _persist_residuals(
    ticker: str,
    log_ret: pd.Series,
    std_resid: pd.Series,
    cond_vol: pd.Series,
    engine: Engine,
) -> None:
    # Align all series on trade_date
    df = (
        pd.DataFrame({
            "log_return": log_ret,
            "std_residual": std_resid,
            "cond_vol": cond_vol,
        })
        .dropna()
        .reset_index()
        .rename(columns={"index": "trade_date", "trade_date": "trade_date"})
    )
    df["ticker"] = ticker

    if df.empty:
        return

    # Ensure trade_date column is named correctly after reset_index
    if "trade_date" not in df.columns and df.columns[0] != "ticker":
        df = df.rename(columns={df.columns[0]: "trade_date"})

    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, df[["ticker", "trade_date", "log_return", "std_residual", "cond_vol"]].to_dict("records"))


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run_econometrics(
    tickers: Optional[List[str]] = None,
    engine: Optional[Engine] = None,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    Run Phase 2 for all tickers.

    Returns
    -------
    residuals   : {ticker: standardised_residuals}  — input for Copula phase
    log_returns : wide DataFrame of log-returns     — input for Simulation phase
    """
    tickers = tickers or PORTFOLIO.tickers
    engine = engine or get_engine()

    log_returns = load_log_returns(tickers, engine)

    adf_results: List[dict] = []
    residuals: Dict[str, pd.Series] = {}

    for ticker in tickers:
        if ticker not in log_returns.columns:
            logger.warning("Skipping %s — no data in warehouse", ticker)
            continue

        series = log_returns[ticker].dropna()

        # 1. Stationarity gate
        adf_results.append(adf_test(series, ticker))

        # 2. EGARCH fitting
        _, std_resid, cond_vol = fit_egarch(series, ticker)

        # Align index lengths (AR(1) mean model loses one obs)
        common = series.index.intersection(std_resid.index)
        std_resid = std_resid.loc[common]
        cond_vol = cond_vol.loc[common]
        log_ret_aligned = series.loc[common]

        residuals[ticker] = std_resid

        # 3. Persist
        _persist_residuals(ticker, log_ret_aligned, std_resid, cond_vol, engine)

    n_nonstationary = sum(not r["is_stationary"] for r in adf_results)
    if n_nonstationary:
        logger.warning("%d series failed ADF stationarity test", n_nonstationary)

    return residuals, log_returns
