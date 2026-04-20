"""
ALPHAchef — Page 1: Volatility Surface

Displays EGARCH(1,1) conditional volatility outputs:
  • 3-D volatility surface across assets × time
  • Standardised residuals heatmap (leverage-effect signature)
  • Per-asset conditional volatility time-series
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.charts import (
    cond_vol_lines,
    egarch_residuals_heatmap,
    vol_surface_3d,
)
from dashboard.utils.db import has_data, load_egarch_residuals
from src.config import PORTFOLIO

st.set_page_config(page_title="Volatility Surface — ALPHAchef", layout="wide")

st.title("📊 EGARCH Volatility Surface")
st.caption(
    "EGARCH(1,1) with Student-t innovations — captures asymmetric leverage effect: "
    "negative shocks inflate volatility more than equivalent positive shocks."
)

if not has_data("egarch_residuals"):
    st.info("No EGARCH data found. Run the pipeline from the main page first.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    selected = st.multiselect(
        "Assets", PORTFOLIO.tickers, default=PORTFOLIO.tickers[:6]
    )
    date_range = st.slider(
        "Year range",
        min_value=2010, max_value=2025, value=(2015, 2025)
    )

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_egarch_residuals(selected or PORTFOLIO.tickers)
df = df[
    (df["trade_date"].dt.year >= date_range[0]) &
    (df["trade_date"].dt.year <= date_range[1])
]

if df.empty:
    st.warning("No data in selected range.")
    st.stop()

# ── Metrics row ────────────────────────────────────────────────────────────────
cols = st.columns(len(selected) if selected else len(PORTFOLIO.tickers))
for i, ticker in enumerate((selected or PORTFOLIO.tickers)):
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        continue
    avg_vol = sub["cond_vol"].mean() * 100
    max_vol = sub["cond_vol"].max() * 100
    cols[i].metric(ticker, f"{avg_vol:.2f}%", f"peak {max_vol:.1f}%")

st.divider()

# ── 3-D surface ────────────────────────────────────────────────────────────────
st.subheader("3-D Conditional Volatility Surface")
st.plotly_chart(vol_surface_3d(df), use_container_width=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Conditional Volatility Time-Series")
    st.plotly_chart(cond_vol_lines(df, selected or None), use_container_width=True)

with right:
    st.subheader("Standardised Residuals Heatmap")
    st.plotly_chart(egarch_residuals_heatmap(df), use_container_width=True)

# ── Descriptive statistics ─────────────────────────────────────────────────────
st.subheader("EGARCH Residual Statistics")
stats_df = (
    df.groupby("ticker")[["std_residual", "cond_vol"]]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)
stats_df.columns = [
    "Resid μ", "Resid σ", "Resid min", "Resid max",
    "Vol μ (%)", "Vol σ (%)", "Vol min", "Vol max"
]
for c in ["Vol μ (%)", "Vol σ (%)", "Vol min", "Vol max"]:
    stats_df[c] = (stats_df[c] * 100).round(3)
st.dataframe(stats_df, use_container_width=True)
