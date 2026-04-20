"""
Phase 3 — Copula Calibration

Steps:
  1. Probability Integral Transform (PIT) → uniform margins  U ∈ (0,1)^d
  2. Fit multivariate Student-t copula via MLE  (correlation matrix R, dof ν)
  3. Sample correlated shocks to feed the simulation engine

Mathematical basis — Sklar's theorem:
  F(x₁,...,x_d) = C(F₁(x₁),...,F_d(x_d))

Student-t copula density:
  c_t(u; R, ν) = f_t(t_ν⁻¹(u₁),...,t_ν⁻¹(u_d); R, ν) / ∏ f_t(t_ν⁻¹(u_i); ν)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

logger = logging.getLogger(__name__)


# ── Step 1 — Probability Integral Transform ────────────────────────────────────

def probability_integral_transform(
    residuals: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Rank-based PIT: maps standardised EGARCH residuals to uniform [0,1] margins.

    Dividing by (n+1) ensures strict membership in (0,1), avoiding boundary
    issues when we later apply the inverse t-CDF.
    """
    aligned = pd.DataFrame(residuals).dropna()
    n = len(aligned)
    uniforms = aligned.rank() / (n + 1)
    logger.info("PIT: %d obs × %d assets → uniform margins", n, aligned.shape[1])
    return uniforms


# ── Step 2 — Student-t copula MLE ─────────────────────────────────────────────

def _build_corr_matrix(rho_vec: np.ndarray, d: int) -> np.ndarray:
    """Reconstruct symmetric correlation matrix from lower-triangular vector."""
    R = np.eye(d)
    idx = np.tril_indices(d, -1)
    R[idx] = rho_vec
    R = R + R.T - np.eye(d)
    return R


def _student_t_copula_nll(params: np.ndarray, u: np.ndarray) -> float:
    """
    Negative log-likelihood of the multivariate Student-t copula.

    params layout:  [rho_lower_tri (n_corr values)  |  log(ν - 2)]
    This ensures ν > 2 (finite variance) throughout optimisation.
    """
    n_obs, d = u.shape
    n_corr = d * (d - 1) // 2

    rho_vec = params[:n_corr]
    nu = float(np.exp(params[n_corr]) + 2.0)   # ν ∈ (2, ∞)

    R = _build_corr_matrix(rho_vec, d)

    # Positive-definiteness check
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() <= 1e-8:
        return 1e12

    sign, log_det_R = np.linalg.slogdet(R)
    if sign <= 0:
        return 1e12

    R_inv = np.linalg.inv(R)

    # Transform uniform margins to t-quantiles
    x = stats.t.ppf(u, df=nu)           # (n_obs, d)

    # Mahalanobis distance under R
    mahal = np.einsum("ni,ij,nj->n", x, R_inv, x)   # (n_obs,)

    # Multivariate t log-density
    log_mvt = (
        gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - (d / 2.0) * np.log(nu * np.pi)
        - 0.5 * log_det_R
        - ((nu + d) / 2.0) * np.log1p(mahal / nu)
    )

    # Subtract univariate t marginal log-densities (copula density formula)
    log_marginals = stats.t.logpdf(x, df=nu).sum(axis=1)   # (n_obs,)

    nll = -np.sum(log_mvt - log_marginals)
    return nll if np.isfinite(nll) else 1e12


def _kendall_to_pearson(u: np.ndarray) -> np.ndarray:
    """
    Initialise correlation matrix via Kendall's τ → Pearson ρ transform:
        ρ = sin(π τ / 2)
    This is exact for elliptical distributions.
    """
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = stats.kendalltau(u[:, i], u[:, j])
            rho = np.clip(np.sin(np.pi * tau / 2.0), -0.99, 0.99)
            R[i, j] = R[j, i] = rho
    return R


def fit_student_t_copula(
    uniform_margins: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """
    Fit multivariate Student-t copula by maximum likelihood.

    Returns
    -------
    R  : (d×d) fitted correlation matrix
    nu : fitted degrees-of-freedom  (ν > 2)
    """
    u = uniform_margins.values.astype(np.float64)
    d = u.shape[1]
    n_corr = d * (d - 1) // 2

    # Warm-start: Kendall-tau correlation + ν = 5
    R_init = _kendall_to_pearson(u)
    idx = np.tril_indices(d, -1)
    params0 = np.concatenate([R_init[idx], [np.log(3.0)]])  # log(5-2)

    bounds = [(-0.99, 0.99)] * n_corr + [(None, None)]

    logger.info("Fitting Student-t copula: d=%d  n_params=%d", d, len(params0))
    result = optimize.minimize(
        _student_t_copula_nll,
        params0,
        args=(u,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )

    if not result.success:
        logger.warning("Copula optimisation: %s", result.message)

    rho_fitted = result.x[:n_corr]
    nu_fitted = float(np.exp(result.x[n_corr]) + 2.0)

    R_fitted = _build_corr_matrix(rho_fitted, d)
    logger.info(
        "Copula fitted: ν=%.2f  min_eigval=%.4f  nll=%.2f",
        nu_fitted, np.linalg.eigvalsh(R_fitted).min(), result.fun,
    )
    return R_fitted, nu_fitted


# ── Step 3 — Correlated shock sampling ────────────────────────────────────────

def sample_copula(
    R: np.ndarray,
    nu: float,
    n_samples: int,
) -> np.ndarray:
    """
    Draw (n_samples, d) samples from the multivariate Student-t copula.

    Algorithm (Cholesky):
      1. Z  ~ N(0, I_d)
      2. W  = L Z  where  LL^T = R   (correlated normals)
      3. χ² ~ χ²(ν)
      4. T  = W / sqrt(χ²/ν)          (multivariate t)
      5. U  = F_t(T; ν)               (uniform margins via t-CDF)
    """
    d = R.shape[0]
    L = np.linalg.cholesky(R)

    Z = np.random.standard_normal((n_samples, d))
    W = Z @ L.T

    chi2 = np.random.chisquare(nu, size=n_samples)
    T = W / np.sqrt(chi2[:, None] / nu)

    U = stats.t.cdf(T, df=nu)
    return U  # shape: (n_samples, d)


def copula_uniforms_to_normals(U: np.ndarray) -> np.ndarray:
    """
    Transform copula uniform samples back to standard normals via Φ⁻¹.
    These become the correlated dW_t increments in the SDE.
    """
    return stats.norm.ppf(np.clip(U, 1e-7, 1 - 1e-7))
