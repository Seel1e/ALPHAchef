"""
Phase 5 (partial) — Risk Metrics

Computes institutional-grade tail-risk measures from simulated terminal prices:
    • VaR (Value at Risk)          — worst loss not exceeded at p% confidence
    • CVaR / Expected Shortfall    — mean loss conditional on exceeding VaR
    • Full distributional statistics
    • Persistence to simulation_results table
"""

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import MODEL, PORTFOLIO

logger = logging.getLogger(__name__)


# ── Portfolio aggregation ──────────────────────────────────────────────────────

def portfolio_returns(
    terminal_prices: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute equal-weighted (or custom) portfolio simulated returns.

    All assets are normalised to S₀ = 100, so return = (S_T - 100) / 100.
    """
    tickers = list(terminal_prices.keys())
    if weights is None:
        w = 1.0 / len(tickers)
        weights = {t: w for t in tickers}

    price_matrix = np.column_stack([terminal_prices[t] for t in tickers])
    weight_vec = np.array([weights[t] for t in tickers])

    portfolio_terminal = price_matrix @ weight_vec           # shape: (n_sims,)
    return (portfolio_terminal - 100.0) / 100.0


# ── Core risk measures ─────────────────────────────────────────────────────────

def var(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical-simulation VaR (positive number = loss)."""
    return float(-np.percentile(returns, (1.0 - confidence) * 100.0))


def cvar(returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    Expected Shortfall / CVaR — mean loss in the tail beyond VaR.
    Coherent risk measure; always ≥ VaR.
    """
    threshold = -var(returns, confidence)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else var(returns, confidence)


# ── Full risk report ───────────────────────────────────────────────────────────

def risk_report(
    terminal_prices: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    confidence: float = 0.99,
    copula_dof: Optional[float] = None,
    tickers: Optional[List[str]] = None,
) -> dict:
    """
    Compile a complete institutional risk report.

    Returns a flat dict suitable for DB persistence and dashboard display.
    """
    ret = portfolio_returns(terminal_prices, weights)
    s = pd.Series(ret)

    report = {
        "run_id": str(uuid.uuid4()),
        "horizon_days": MODEL.horizon_days,
        "n_simulations": len(ret),
        "var_99": var(ret, confidence),
        "cvar_99": cvar(ret, confidence),
        "median_return": float(np.median(ret)),
        "mean_return": float(ret.mean()),
        "std_return": float(ret.std()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "pct_loss_gt10": float((ret < -0.10).mean()),
        "pct_loss_gt20": float((ret < -0.20).mean()),
        "copula_dof": copula_dof,
        "tickers": tickers or list(terminal_prices.keys()),
        "raw_returns": ret,   # not persisted; used for charting
    }

    logger.info(
        "Risk report  VaR₉₉=%.2f%%  CVaR₉₉=%.2f%%  skew=%.3f  kurt=%.3f",
        report["var_99"] * 100,
        report["cvar_99"] * 100,
        report["skewness"],
        report["kurtosis"],
    )
    return report


# ── Persistence ────────────────────────────────────────────────────────────────

_INSERT_SQL = text("""
    INSERT INTO simulation_results (
        run_id, horizon_days, n_simulations,
        var_99, cvar_99, median_return, mean_return, std_return,
        skewness, kurtosis, pct_loss_gt10, pct_loss_gt20,
        copula_dof, tickers
    ) VALUES (
        :run_id, :horizon_days, :n_simulations,
        :var_99, :cvar_99, :median_return, :mean_return, :std_return,
        :skewness, :kurtosis, :pct_loss_gt10, :pct_loss_gt20,
        :copula_dof, :tickers
    )
""")


def persist_report(report: dict, engine: Engine) -> str:
    row = {k: v for k, v in report.items() if k != "raw_returns"}
    with engine.begin() as conn:
        conn.execute(_INSERT_SQL, row)
    logger.info("Simulation run %s persisted.", report["run_id"])
    return report["run_id"]


# ── Per-asset statistics (for dashboard) ──────────────────────────────────────

def per_asset_stats(terminal_prices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Return a DataFrame with per-asset risk statistics."""
    rows = []
    for ticker, prices in terminal_prices.items():
        ret = (prices - 100.0) / 100.0
        rows.append({
            "ticker": ticker,
            "mean_return": ret.mean(),
            "std_return": ret.std(),
            "var_99": var(ret),
            "cvar_99": cvar(ret),
            "prob_loss": (ret < 0).mean(),
        })
    return pd.DataFrame(rows).set_index("ticker")
