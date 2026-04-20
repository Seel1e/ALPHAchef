"""
ALPHAchef — Page 3: Monte Carlo Simulation

Shows results from the most recent simulation run, or triggers a new one:
  • 99% VaR / CVaR with histogram
  • Per-asset risk metrics
  • Simulated terminal price distributions (fan chart)
  • Simulation history comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.charts import (
    per_asset_var_bar,
    return_distribution,
    fan_chart,
)
from dashboard.utils.db import has_data, load_latest_simulation
from src.config import MODEL, PORTFOLIO

st.set_page_config(page_title="Monte Carlo — ALPHAchef", layout="wide")

st.title("🎲 Monte Carlo Simulation Results")
st.caption(
    "Merton Jump-Diffusion SDE:  dS = (μ−λk)S dt + σS dW_t + S(Y−1)dN_t  |  "
    f"100k paths · {MODEL.horizon_days}-day horizon · Student-t copula shocks"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Controls")
    n_sims = st.select_slider("Paths", [1_000, 5_000, 10_000, 50_000, 100_000], value=10_000)
    use_spark = st.toggle("Use PySpark", value=False,
                          help="Distributes across all CPU cores. Slower startup, faster at 100k+")
    run_sim = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Run new simulation ─────────────────────────────────────────────────────────
if run_sim:
    with st.spinner("Loading residuals & fitting copula…"):
        from src.econometrics.egarch import run_econometrics
        from src.copula.calibration import (
            fit_student_t_copula, probability_integral_transform, sample_copula,
        )
        residuals, log_returns = run_econometrics()
        uniforms = probability_integral_transform(residuals)
        R, nu = fit_student_t_copula(uniforms)

    with st.spinner(f"Running {n_sims:,} Monte Carlo paths…"):
        from src.simulation.jump_diffusion import estimate_params, run_simulation
        params = estimate_params(log_returns)
        terminal = run_simulation(
            params, R, nu,
            tickers=PORTFOLIO.tickers,
            use_spark=use_spark,
        )

    with st.spinner("Computing risk metrics…"):
        from src.etl.extractor import get_engine
        from src.risk.metrics import persist_report, per_asset_stats, risk_report
        report = risk_report(terminal, copula_dof=nu, tickers=PORTFOLIO.tickers)
        engine = get_engine()
        persist_report(report, engine)

        st.session_state["mc_report"] = report
        st.session_state["mc_terminal"] = terminal
        st.session_state["mc_asset_stats"] = per_asset_stats(terminal)

    st.cache_data.clear()
    st.success(f"Done!  VaR₉₉ = {report['var_99']:.2%}  |  CVaR₉₉ = {report['cvar_99']:.2%}")


# ── Pull cached session or DB results ─────────────────────────────────────────
report = st.session_state.get("mc_report")
terminal = st.session_state.get("mc_terminal")
asset_stats = st.session_state.get("mc_asset_stats")

if not has_data("simulation_results") and report is None:
    st.info("No simulation results yet. Click **Run Simulation** in the sidebar.")
    st.stop()


# ── Portfolio KPIs ─────────────────────────────────────────────────────────────
sims_df = load_latest_simulation()

if report:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("VaR₉₉",           f"{report['var_99']:.2%}")
    c2.metric("CVaR₉₉",          f"{report['cvar_99']:.2%}")
    c3.metric("Median Return",    f"{report['median_return']:.2%}")
    c4.metric("Skewness",         f"{report['skewness']:.3f}")
    c5.metric("Excess Kurtosis",  f"{report['kurtosis']:.2f}")
    c6.metric("P(loss > 20%)",    f"{report['pct_loss_gt20']:.2%}")
elif not sims_df.empty:
    r = sims_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR₉₉",        f"{r['var_99']:.2%}")
    c2.metric("CVaR₉₉",       f"{r['cvar_99']:.2%}")
    c3.metric("Median Return", f"{r['median_return']:.2%}")
    c4.metric("Copula ν",      f"{r['copula_dof']:.1f}")

st.divider()

# ── Return distribution chart ──────────────────────────────────────────────────
if report and terminal:
    from src.risk.metrics import portfolio_returns
    ret = portfolio_returns(terminal)

    st.subheader("Portfolio Return Distribution")
    st.plotly_chart(
        return_distribution(ret, report["var_99"], report["cvar_99"]),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Terminal Price Distribution (Fan Chart)")
        st.plotly_chart(fan_chart(terminal), use_container_width=True)

    with right:
        st.subheader("Per-Asset VaR₉₉ vs CVaR₉₉")
        if asset_stats is not None:
            st.plotly_chart(per_asset_var_bar(asset_stats), use_container_width=True)

    # Per-asset table
    st.subheader("Per-Asset Risk Summary")
    if asset_stats is not None:
        display = asset_stats.copy()
        for c in ["mean_return", "std_return", "var_99", "cvar_99", "prob_loss"]:
            display[c] = display[c].map("{:.2%}".format)
        st.dataframe(display, use_container_width=True)

    # QQ-plot vs normal
    st.subheader("Return Distribution Q-Q Plot (vs. Normal)")
    from scipy import stats as _stats
    (osm, osr), (slope, intercept, _) = _stats.probplot(ret, dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color="#00d4ff", size=3, opacity=0.6), name="Simulated",
    ))
    fig_qq.add_trace(go.Scatter(
        x=[min(osm), max(osm)],
        y=[slope * min(osm) + intercept, slope * max(osm) + intercept],
        mode="lines", line=dict(color="orange", dash="dash"), name="Normal",
    ))
    fig_qq.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_qq, use_container_width=True)

# ── Historical simulation run comparison ──────────────────────────────────────
if not sims_df.empty:
    st.subheader("Historical Simulation Runs")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=sims_df["run_ts"], y=sims_df["var_99"] * 100,
        mode="lines+markers", name="VaR₉₉ (%)", line=dict(color="#ffd93d"),
    ))
    fig_hist.add_trace(go.Scatter(
        x=sims_df["run_ts"], y=sims_df["cvar_99"] * 100,
        mode="lines+markers", name="CVaR₉₉ (%)", line=dict(color="#ff6b6b"),
    ))
    fig_hist.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="Run Time", yaxis_title="Loss (%)",
        legend=dict(orientation="h", y=-0.3),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
