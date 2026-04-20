"""
ALPHAchef — Streamlit Dashboard
Main landing page: portfolio overview + pipeline controls.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.db import has_data, load_latest_simulation, load_prices
from src.config import PORTFOLIO

st.set_page_config(
    page_title="ALPHAchef",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom dark theme override ─────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
  .stPlotlyChart { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ALPHAchef")
    st.caption("Copula-Based Regime-Switching Jump-Diffusion Forecaster")
    st.divider()
    st.markdown("**Pages**")
    st.page_link("pages/1_Volatility_Surface.py",  label="📊 Volatility Surface")
    st.page_link("pages/2_Copula_Analysis.py",     label="🔗 Copula Analysis")
    st.page_link("pages/3_Monte_Carlo.py",          label="🎲 Monte Carlo")
    st.divider()

    run_pipeline = st.button("▶  Run Full Pipeline", type="primary", use_container_width=True)
    run_etl_only = st.button("   ETL Only",           use_container_width=True)
    st.caption("Runs pipeline directly (no Airflow needed).")


# ── Pipeline trigger ───────────────────────────────────────────────────────────
if run_etl_only:
    with st.spinner("Running ETL…"):
        from src.etl.extractor import run_etl
        summary = run_etl(incremental=True)
    total = sum(r["rows"] for r in summary.values())
    st.success(f"ETL complete — {total} rows upserted across {len(summary)} tickers.")
    st.cache_data.clear()

if run_pipeline:
    with st.spinner("Phase 1/5 — ETL…"):
        from src.etl.extractor import run_etl
        run_etl(incremental=True)
        st.cache_data.clear()

    with st.spinner("Phase 2/5 — EGARCH econometrics…"):
        from src.econometrics.egarch import run_econometrics
        residuals, log_returns = run_econometrics()

    with st.spinner("Phase 3/5 — Copula calibration…"):
        from src.copula.calibration import (
            fit_student_t_copula, probability_integral_transform, sample_copula,
        )
        from src.config import MODEL
        uniforms = probability_integral_transform(residuals)
        R, nu = fit_student_t_copula(uniforms)
        shocks = sample_copula(R, nu, MODEL.n_simulations)

    with st.spinner("Phase 4/5 — Monte Carlo simulation (NumPy)…"):
        from src.simulation.jump_diffusion import estimate_params, run_simulation
        params = estimate_params(log_returns)
        terminal = run_simulation(params, R, nu, use_spark=False)

    with st.spinner("Phase 5/5 — Risk metrics…"):
        from src.etl.extractor import get_engine
        from src.risk.metrics import persist_report, risk_report
        report = risk_report(terminal, copula_dof=nu, tickers=PORTFOLIO.tickers)
        engine = get_engine()
        persist_report(report, engine)
        st.session_state["latest_report"] = report
        st.session_state["latest_uniforms"] = uniforms
        st.session_state["latest_R"] = R
        st.session_state["latest_nu"] = nu
        st.session_state["latest_terminal"] = terminal

    st.cache_data.clear()
    st.success(
        f"Pipeline complete!  VaR₉₉ = {report['var_99']:.2%}  |  "
        f"CVaR₉₉ = {report['cvar_99']:.2%}"
    )


# ── Main content ───────────────────────────────────────────────────────────────
st.title("ALPHAchef  —  Quantitative Risk Dashboard")
st.caption(
    "Merton Jump-Diffusion · EGARCH(1,1) · Multivariate Student-t Copula · "
    "100k Monte Carlo paths · 99% VaR / CVaR"
)

if not has_data():
    st.info("No data in database yet. Click **Run Full Pipeline** in the sidebar to get started.")
    st.stop()


# ── KPI tiles ──────────────────────────────────────────────────────────────────
sims_df = load_latest_simulation()
col1, col2, col3, col4, col5 = st.columns(5)

if not sims_df.empty:
    latest = sims_df.iloc[0]
    col1.metric("VaR₉₉", f"{latest['var_99']:.2%}")
    col2.metric("CVaR₉₉", f"{latest['cvar_99']:.2%}")
    col3.metric("Median Return", f"{latest['median_return']:.2%}")
    col4.metric("Copula DoF (ν)", f"{latest['copula_dof']:.1f}")
    col5.metric("Simulations", f"{latest['n_simulations']:,.0f}")
else:
    st.info("No simulation results yet. Run the full pipeline.")


st.divider()


# ── Historical price chart ─────────────────────────────────────────────────────
st.subheader("Historical Adjusted Close Prices")
prices_df = load_prices(PORTFOLIO.tickers)
if not prices_df.empty:
    wide = prices_df.pivot(index="trade_date", columns="ticker", values="adj_close")
    normalised = (wide / wide.iloc[0]) * 100   # rebased to 100

    fig = go.Figure()
    palette = [
        "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
        "#845ec2", "#f9a03f", "#0089ba", "#c34b4b",
    ]
    for i, col in enumerate(normalised.columns):
        fig.add_trace(go.Scatter(
            x=normalised.index, y=normalised[col],
            name=col, line=dict(color=palette[i % len(palette)], width=1.5),
        ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Rebased Price (100 = start)",
        legend=dict(orientation="h", y=-0.2),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Simulation history table ───────────────────────────────────────────────────
if not sims_df.empty:
    st.subheader("Simulation Run History")
    display_cols = ["run_ts", "var_99", "cvar_99", "median_return", "copula_dof", "n_simulations"]
    display_df = sims_df[display_cols].copy()
    display_df["run_ts"] = display_df["run_ts"].dt.strftime("%Y-%m-%d %H:%M")
    for c in ["var_99", "cvar_99", "median_return"]:
        display_df[c] = display_df[c].map("{:.2%}".format)
    st.dataframe(display_df, use_container_width=True)
