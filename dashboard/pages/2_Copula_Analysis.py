"""
ALPHAchef — Page 2: Copula Analysis

Fits the multivariate Student-t copula from EGARCH residuals on-demand
and displays:
  • Fitted correlation matrix heatmap
  • Pairwise copula scatter / contour density
  • Tail-dependence coefficients
  • Degrees-of-freedom indicator (lower ν → fatter tails → more crash coupling)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.charts import (
    copula_contour,
    copula_scatter_matrix,
    correlation_heatmap,
)
from dashboard.utils.db import has_data, load_egarch_residuals
from src.config import PORTFOLIO

st.set_page_config(page_title="Copula Analysis — ALPHAchef", layout="wide")

st.title("🔗 Copula Dependency Analysis")
st.caption(
    "Sklar's theorem: F(x₁,…,xd) = C(F₁(x₁),…,Fd(xd)).  "
    "Fitting a multivariate Student-t copula reveals non-linear tail dependence — "
    "how assets co-crash during extreme events."
)

if not has_data("egarch_residuals"):
    st.info("No EGARCH residuals found. Run the pipeline from the main page first.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    selected = st.multiselect("Assets", PORTFOLIO.tickers, default=PORTFOLIO.tickers[:5])
    pair_a = st.selectbox("Contour: Asset A", selected or PORTFOLIO.tickers[:2], index=0)
    pair_b = st.selectbox("Contour: Asset B", selected or PORTFOLIO.tickers[:2], index=1)
    fit_now = st.button("🔄 Re-fit Copula", type="primary", use_container_width=True)

# ── Load residuals & compute uniforms ─────────────────────────────────────────
tickers = selected or PORTFOLIO.tickers
df = load_egarch_residuals(tickers)

@st.cache_data(show_spinner="Applying PIT…")
def get_uniforms(tickers_key: str):
    from src.copula.calibration import probability_integral_transform
    from src.econometrics.egarch import run_econometrics
    residuals, _ = run_econometrics(list(tickers_key.split(",")))
    return probability_integral_transform(residuals)

@st.cache_data(show_spinner="Fitting Student-t copula…", ttl=3600)
def get_copula_params(tickers_key: str):
    from src.copula.calibration import fit_student_t_copula
    uniforms = get_uniforms(tickers_key)
    R, nu = fit_student_t_copula(uniforms)
    return R, nu, uniforms

cache_key = ",".join(sorted(tickers))
if fit_now:
    st.cache_data.clear()

R, nu, uniforms = get_copula_params(cache_key)
uniforms_df = pd.DataFrame(uniforms)   # tickers as columns

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Degrees of Freedom (ν)", f"{nu:.2f}",
          help="Lower ν → heavier tails → stronger crash coupling")
k2.metric("Assets modelled", len(tickers))

# Tail dependence coefficient for Student-t copula:  λ = 2 * t_ν+1(−√((ν+1)(1−ρ)/(1+ρ)))
from scipy import stats as _stats
avg_rho = float(np.mean(R[np.tril_indices(len(tickers), -1)]))
if nu > 2:
    lambda_tail = 2.0 * _stats.t.cdf(
        -np.sqrt((nu + 1) * (1 - avg_rho) / (1 + avg_rho + 1e-9)),
        df=nu + 1,
    )
else:
    lambda_tail = float("nan")
k3.metric("Avg Tail-Dep. Coeff. (λ)", f"{lambda_tail:.3f}",
          help="Probability both assets crash simultaneously in the tail")

st.divider()

# ── Correlation matrix ─────────────────────────────────────────────────────────
st.subheader("Fitted Correlation Matrix")
st.plotly_chart(correlation_heatmap(R, tickers), use_container_width=True)

st.divider()

# ── Two-column: contour + scatter ─────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader(f"Copula Density Contour — {pair_a} vs {pair_b}")
    if pair_a in uniforms_df.columns and pair_b in uniforms_df.columns:
        st.plotly_chart(copula_contour(uniforms_df, pair_a, pair_b), use_container_width=True)
    else:
        st.warning("Select two different assets in the sidebar.")

with right:
    st.subheader("Upper-Tail Co-exceedance")
    # Show % of days where BOTH assets exceed their 95th percentile simultaneously
    if not uniforms_df.empty and len(uniforms_df.columns) >= 2:
        tail_threshold = 0.95
        rows = []
        cols_list = uniforms_df.columns.tolist()
        for i, t1 in enumerate(cols_list):
            for j, t2 in enumerate(cols_list):
                if j <= i:
                    continue
                joint_tail = ((uniforms_df[t1] > tail_threshold) &
                              (uniforms_df[t2] > tail_threshold)).mean()
                rows.append({"Asset A": t1, "Asset B": t2,
                             "Joint Tail (>95th %)": f"{joint_tail:.3%}"})
        tail_df = pd.DataFrame(rows)
        st.dataframe(tail_df, use_container_width=True, height=300)

# ── Full pairwise scatter matrix ───────────────────────────────────────────────
with st.expander("Pairwise Uniform-Margin Scatter Matrix (all assets)"):
    if len(uniforms_df.columns) <= 8:
        st.plotly_chart(copula_scatter_matrix(uniforms_df), use_container_width=True)
    else:
        st.info("Select ≤ 8 assets for the scatter matrix.")
