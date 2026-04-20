"""
Phase 4 — Merton Jump-Diffusion Monte Carlo Simulation

Implements the Merton (1976) SDE:

    dS_t = (μ - λk) S_t dt  +  σ S_t dW_t  +  S_t (Y_t - 1) dN_t

where:
    μ        : annualised drift
    σ        : diffusion (EGARCH-estimated) volatility
    λ        : Poisson jump intensity  (jumps per year)
    k        : E[Y-1] = exp(μ_J + ½σ_J²) - 1   (compensator term)
    dW_t     : Wiener increment — drawn from Student-t copula (correlated)
    Y_t      : log-normal jump size  Y_t = exp(μ_J + σ_J Z_J),  Z_J ~ N(0,1)
    dN_t     : Poisson counter  N_t ~ Poisson(λ dt)

Euler-Maruyama discretisation (log-price form, exact for geometric BM):

    log(S_{t+Δt}/S_t) = (μ - λk - ½σ²)Δt  +  σ√Δt W_t
                       + Σ_{i=1}^{N(Δt)} log(Y_i)

The copula's correlated standard-normal shocks drive W_t, ensuring
synchronised crashes across assets in the portfolio.

Execution strategy:
    • Pure-NumPy  : fast, runs on any machine
    • PySpark     : distributes across CPU cores for 100k+ paths
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MODEL, PORTFOLIO, SPARK
from src.copula.calibration import copula_uniforms_to_normals, sample_copula

logger = logging.getLogger(__name__)


# ── Parameter estimation ───────────────────────────────────────────────────────

def estimate_params(log_returns: pd.DataFrame) -> Dict[str, dict]:
    """
    Method-of-moments estimator for the continuous diffusion component.
    Jump parameters come from ModelConfig (calibrated externally or set as priors).
    """
    params: Dict[str, dict] = {}
    dt = 1 / 252

    for ticker in log_returns.columns:
        ret = log_returns[ticker].dropna()
        lam = MODEL.jump_lambda
        mu_j = MODEL.jump_mu
        sig_j = MODEL.jump_sigma
        k = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0

        # Subtract expected jump contribution from observed variance/mean
        total_var = ret.var() * 252
        jump_var = lam * (sig_j ** 2 + mu_j ** 2)
        sigma = max(np.sqrt(max(total_var - jump_var, 1e-8)), 0.01)

        mu = ret.mean() * 252 + 0.5 * sigma ** 2 + lam * k

        params[ticker] = {
            "mu": mu,
            "sigma": sigma,
            "lambda": lam,
            "mu_j": mu_j,
            "sigma_j": sig_j,
            "k": k,
        }
        logger.info("Params %-6s  μ=%.4f  σ=%.4f  λ=%.1f", ticker, mu, sigma, lam)

    return params


# ── Core SDE kernel (NumPy vectorised) ────────────────────────────────────────

def _merton_log_returns(
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sig_j: float,
    k: float,
    corr_normals: np.ndarray,  # (n_sims, n_days)  ← from copula
    dt: float = 1 / 252,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Vectorised Euler-Maruyama step for Merton SDE.

    Returns log-return increments of shape (n_sims, n_days).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sims, n_days = corr_normals.shape

    # ── Continuous diffusion term ─────────────────────────────────────────
    drift = (mu - lam * k - 0.5 * sigma ** 2) * dt              # scalar
    diffusion = sigma * np.sqrt(dt) * corr_normals               # (n_sims, n_days)

    # ── Jump term  Σ log(Y_i)  where N ~ Poisson(λ dt) ───────────────────
    N_jumps = rng.poisson(lam * dt, size=(n_sims, n_days))        # counts
    max_j = max(int(N_jumps.max()), 1)

    # Draw all potential jump sizes at once, then mask by actual count
    jump_log_sizes = rng.normal(mu_j, sig_j, size=(n_sims, n_days, max_j))
    mask = np.arange(max_j)[None, None, :] < N_jumps[:, :, None]
    total_log_jump = (jump_log_sizes * mask).sum(axis=2)           # (n_sims, n_days)

    return drift + diffusion + total_log_jump


def _terminal_price(log_ret_increments: np.ndarray, s0: float = 100.0) -> np.ndarray:
    """Convert log-return increments to terminal prices (S_T = S_0 * exp(Σ r_t))."""
    return s0 * np.exp(log_ret_increments.sum(axis=1))


# ── All-day correlated shock generation ───────────────────────────────────────

def _generate_chunk_shocks(
    L: np.ndarray,
    nu: float,
    chunk_size: int,
    n_days: int,
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate copula-correlated shocks for one chunk of simulations.

    Peak memory: chunk_size × n_days × n_assets × ~5 arrays × 8 bytes.
    With chunk_size=5000, n_days=252, n_assets=8: ~160 MB peak — safe on 8 GB RAM.
    """
    total = chunk_size * n_days
    Z = rng.standard_normal((total, n_assets)) @ L.T          # correlated normals
    chi2 = rng.chisquare(nu, size=total)
    T = Z / np.sqrt(chi2[:, None] / nu)                       # multivariate t
    del Z, chi2                                                # free immediately
    U = stats.t.cdf(T, df=nu)
    del T
    W = copula_uniforms_to_normals(U)
    del U
    return W.reshape(chunk_size, n_days, n_assets)            # (C, D, A)


# ── NumPy simulation (fallback / local) ───────────────────────────────────────

# Chunk size governs peak RAM: 5000 sims × 252 days × 8 assets ≈ 160 MB
_CHUNK_SIZE = 5_000


def simulate_numpy(
    params: Dict[str, dict],
    R: np.ndarray,
    nu: float,
    tickers: Optional[List[str]] = None,
    n_sims: Optional[int] = None,
    n_days: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Chunked NumPy Monte Carlo — processes _CHUNK_SIZE paths at a time to
    keep peak RAM under ~200 MB regardless of total path count.
    """
    tickers = tickers or list(params.keys())
    n_sims = n_sims or MODEL.n_simulations
    n_days = n_days or MODEL.horizon_days
    n_assets = len(tickers)

    logger.info("NumPy MC: %d sims × %d days × %d assets (chunk=%d)",
                n_sims, n_days, n_assets, _CHUNK_SIZE)

    L = np.linalg.cholesky(R)
    rng = np.random.default_rng()

    # Accumulate terminal prices across chunks
    accum: Dict[str, list] = {t: [] for t in tickers}

    for start in range(0, n_sims, _CHUNK_SIZE):
        chunk = min(_CHUNK_SIZE, n_sims - start)
        W = _generate_chunk_shocks(L, nu, chunk, n_days, n_assets, rng)

        for i, ticker in enumerate(tickers):
            p = params[ticker]
            log_rets = _merton_log_returns(
                p["mu"], p["sigma"], p["lambda"], p["mu_j"], p["sigma_j"], p["k"],
                W[:, :, i],       # (chunk, n_days)
                rng=rng,
            )
            accum[ticker].extend(_terminal_price(log_rets).tolist())

        if (start // _CHUNK_SIZE) % 5 == 0:
            pct = min(start + _CHUNK_SIZE, n_sims) / n_sims * 100
            logger.info("MC progress: %.0f%%  (%d / %d paths)", pct, min(start + _CHUNK_SIZE, n_sims), n_sims)

    results: Dict[str, np.ndarray] = {t: np.array(v) for t, v in accum.items()}
    for ticker in tickers:
        logger.info("%-6s  median terminal = %.2f", ticker, np.median(results[ticker]))
    return results


# ── PySpark simulation (production) ───────────────────────────────────────────

def simulate_spark(
    params: Dict[str, dict],
    R: np.ndarray,
    nu: float,
    tickers: Optional[List[str]] = None,
    n_sims: Optional[int] = None,
    n_days: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    PySpark-distributed Monte Carlo simulation.

    Each partition independently samples from the copula and simulates a
    chunk of paths, then collects terminal prices to the driver.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.warning("PySpark not available, falling back to NumPy simulation")
        return simulate_numpy(params, R, nu, tickers, n_sims, n_days)

    tickers = tickers or list(params.keys())
    n_sims = n_sims or MODEL.n_simulations
    n_days = n_days or MODEL.horizon_days

    spark = (
        SparkSession.builder
        .appName(SPARK.app_name)
        .master(SPARK.master)
        .config("spark.executor.memory", SPARK.executor_memory)
        .config("spark.driver.memory", SPARK.driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Broadcast immutable objects to all workers
    bc_params = spark.sparkContext.broadcast(params)
    bc_R = spark.sparkContext.broadcast(R)
    bc_nu = spark.sparkContext.broadcast(float(nu))
    bc_tickers = spark.sparkContext.broadcast(tickers)
    bc_n_days = spark.sparkContext.broadcast(n_days)

    # Partition simulation IDs across workers
    partition_size = max(500, n_sims // (spark.sparkContext.defaultParallelism * 4))
    chunks = [
        list(range(i, min(i + partition_size, n_sims)))
        for i in range(0, n_sims, partition_size)
    ]
    rdd = spark.sparkContext.parallelize(chunks, numSlices=len(chunks))

    def simulate_chunk(sim_ids: list):
        """Runs inside each Spark partition — no shared state with driver."""
        import numpy as _np
        from scipy import stats as _stats

        _R = bc_R.value
        _nu = bc_nu.value
        _tickers = bc_tickers.value
        _params = bc_params.value
        _n_days = bc_n_days.value
        n = len(sim_ids)
        n_assets = len(_tickers)
        rng = _np.random.default_rng(sim_ids[0])  # reproducible per-partition

        # ── Correlated shocks ──────────────────────────────────────────────
        L = _np.linalg.cholesky(_R)
        total = n * _n_days
        Z = rng.standard_normal((total, n_assets)) @ L.T
        chi2 = rng.chisquare(_nu, size=total)
        T = Z / _np.sqrt(chi2[:, None] / _nu)
        U = _stats.t.cdf(T, df=_nu)
        W_all = _stats.norm.ppf(_np.clip(U, 1e-7, 1 - 1e-7)).reshape(n, _n_days, n_assets)

        # ── Per-asset Merton SDE ───────────────────────────────────────────
        chunk_results = {}
        for i, ticker in enumerate(_tickers):
            p = _params[ticker]
            dt = 1.0 / 252
            lam, mu_j, sig_j, k = p["lambda"], p["mu_j"], p["sigma_j"], p["k"]
            drift = (p["mu"] - lam * k - 0.5 * p["sigma"] ** 2) * dt
            diffusion = p["sigma"] * _np.sqrt(dt) * W_all[:, :, i]

            N_j = rng.poisson(lam * dt, size=(n, _n_days))
            max_j = max(int(N_j.max()), 1)
            jl = rng.normal(mu_j, sig_j, (n, _n_days, max_j))
            mask = _np.arange(max_j)[None, None, :] < N_j[:, :, None]
            total_jl = (jl * mask).sum(axis=2)

            log_rets = drift + diffusion + total_jl
            chunk_results[ticker] = (100.0 * _np.exp(log_rets.sum(axis=1))).tolist()

        # Return list of (ticker, terminal_price) tuples
        rows = []
        for ticker, vals in chunk_results.items():
            rows.extend((ticker, v) for v in vals)
        return rows

    raw = rdd.flatMap(simulate_chunk).collect()
    spark.stop()

    # Reconstruct per-ticker arrays
    results: Dict[str, list] = {t: [] for t in tickers}
    for ticker, val in raw:
        results[ticker].append(val)

    logger.info("Spark MC complete: %d paths per asset", len(results[tickers[0]]))
    return {t: np.array(v) for t, v in results.items()}


# ── Unified entry point ────────────────────────────────────────────────────────

def run_simulation(
    params: Dict[str, dict],
    R: np.ndarray,
    nu: float,
    tickers: Optional[List[str]] = None,
    use_spark: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Execute the Monte Carlo simulation.

    use_spark=True  → PySpark distributed (production)
    use_spark=False → NumPy vectorised   (local / testing)
    """
    if use_spark:
        return simulate_spark(params, R, nu, tickers)
    return simulate_numpy(params, R, nu, tickers)
