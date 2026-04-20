"""Reusable Plotly chart builders for the ALPHAchef dashboard."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


PALETTE = [
    "#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
    "#845ec2", "#f9a03f", "#0089ba", "#c34b4b",
]


# ── Volatility charts ──────────────────────────────────────────────────────────

def vol_surface_3d(residuals_df: pd.DataFrame) -> go.Figure:
    """
    3-D volatility surface: ticker × time → conditional volatility.
    X = time, Y = ticker index, Z = cond_vol.
    """
    pivot = residuals_df.pivot_table(
        index="trade_date", columns="ticker", values="cond_vol"
    ).dropna()

    tickers = list(pivot.columns)
    dates = pivot.index
    Z = pivot.values.T  # shape: (n_tickers, n_dates)

    fig = go.Figure(
        data=[go.Surface(
            x=np.arange(len(dates)),
            y=np.arange(len(tickers)),
            z=Z * 100,
            colorscale="Plasma",
            opacity=0.9,
            colorbar=dict(title="Cond. Vol (%)", len=0.6),
        )]
    )
    fig.update_layout(
        title="3-D Conditional Volatility Surface (EGARCH)",
        scene=dict(
            xaxis=dict(title="Trading Days"),
            yaxis=dict(title="Asset", tickvals=list(range(len(tickers))), ticktext=tickers),
            zaxis=dict(title="Cond. Vol (%)"),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        template="plotly_dark",
        height=600,
    )
    return fig


def egarch_residuals_heatmap(residuals_df: pd.DataFrame) -> go.Figure:
    """Heatmap of standardised residuals across assets and time."""
    pivot = residuals_df.pivot_table(
        index="trade_date", columns="ticker", values="std_residual"
    ).dropna()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values.T,
            x=pivot.index,
            y=list(pivot.columns),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Std Residual"),
        )
    )
    fig.update_layout(
        title="EGARCH Standardised Residuals Heatmap",
        xaxis_title="Date",
        yaxis_title="Asset",
        template="plotly_dark",
        height=400,
    )
    return fig


def cond_vol_lines(residuals_df: pd.DataFrame, tickers: Optional[list] = None) -> go.Figure:
    """Line chart of conditional volatility for selected tickers."""
    tickers = tickers or residuals_df["ticker"].unique().tolist()
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        sub = residuals_df[residuals_df["ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=sub["trade_date"], y=sub["cond_vol"] * 100,
            name=ticker, line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
        ))
    fig.update_layout(
        title="EGARCH Conditional Volatility",
        xaxis_title="Date", yaxis_title="Daily Cond. Vol (%)",
        template="plotly_dark", height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ── Copula charts ──────────────────────────────────────────────────────────────

def copula_scatter_matrix(uniforms_df: pd.DataFrame) -> go.Figure:
    """Pairwise scatter of uniform margins coloured by density."""
    tickers = uniforms_df.columns.tolist()
    n = len(tickers)
    fig = make_subplots(rows=n, cols=n, shared_xaxes=False, shared_yaxes=False)

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i == j:
                fig.add_trace(
                    go.Histogram(x=uniforms_df[ti], nbinsx=40,
                                 showlegend=False, marker_color=PALETTE[i % len(PALETTE)]),
                    row=i + 1, col=j + 1,
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=uniforms_df[tj], y=uniforms_df[ti],
                        mode="markers",
                        marker=dict(size=2, opacity=0.4, color=PALETTE[i % len(PALETTE)]),
                        showlegend=False,
                    ),
                    row=i + 1, col=j + 1,
                )

    fig.update_layout(
        title="Copula Uniform Margins — Pairwise Dependency",
        template="plotly_dark",
        height=max(600, 150 * n),
    )
    return fig


def copula_contour(uniforms_df: pd.DataFrame, asset_a: str, asset_b: str) -> go.Figure:
    """2-D contour density plot for an asset pair — reveals tail co-movement."""
    u = uniforms_df[asset_a].values
    v = uniforms_df[asset_b].values

    # KDE on the uniform margins
    xy = np.vstack([u, v])
    kde = stats.gaussian_kde(xy, bw_method=0.1)

    grid_pts = 80
    ug = np.linspace(0.01, 0.99, grid_pts)
    vg = np.linspace(0.01, 0.99, grid_pts)
    UG, VG = np.meshgrid(ug, vg)
    Z = kde(np.vstack([UG.ravel(), VG.ravel()])).reshape(grid_pts, grid_pts)

    fig = go.Figure(
        data=go.Contour(
            x=ug, y=vg, z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="Density"),
        )
    )
    fig.update_layout(
        title=f"Copula Density — {asset_a} vs {asset_b}",
        xaxis_title=f"U({asset_a})", yaxis_title=f"U({asset_b})",
        template="plotly_dark", height=500,
    )
    return fig


def correlation_heatmap(R: np.ndarray, tickers: list) -> go.Figure:
    """Fitted copula correlation matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=R,
            x=tickers, y=tickers,
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            text=np.round(R, 2),
            texttemplate="%{text}",
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(
        title="Student-t Copula Correlation Matrix",
        template="plotly_dark", height=500,
    )
    return fig


# ── Monte Carlo charts ─────────────────────────────────────────────────────────

def return_distribution(
    returns: np.ndarray,
    var_99: float,
    cvar_99: float,
    title: str = "Portfolio Return Distribution",
) -> go.Figure:
    """Histogram of simulated portfolio returns with VaR/CVaR overlays."""
    fig = go.Figure()

    # Main histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=200,
        name="Simulated returns",
        marker_color="rgba(0, 212, 255, 0.6)",
        marker_line=dict(width=0),
    ))

    # VaR line
    fig.add_vline(
        x=-var_99 * 100,
        line=dict(color="orange", width=2, dash="dash"),
        annotation_text=f"VaR₉₉ = {var_99:.1%}",
        annotation_position="top right",
    )
    # CVaR line
    fig.add_vline(
        x=-cvar_99 * 100,
        line=dict(color="red", width=2, dash="dot"),
        annotation_text=f"CVaR₉₉ = {cvar_99:.1%}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Portfolio Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        bargap=0.02,
        height=450,
        showlegend=False,
    )
    return fig


def fan_chart(
    terminal_prices: Dict[str, np.ndarray],
    n_display_paths: int = 200,
) -> go.Figure:
    """
    Simulated price path fan chart for all assets.
    Shows a random sample of paths + percentile bands.
    """
    fig = go.Figure()

    for i, (ticker, prices) in enumerate(terminal_prices.items()):
        n_sims = len(prices)
        # Percentile bands  (start at 100, end at terminal prices)
        pcts = [5, 25, 50, 75, 95]
        pct_vals = np.percentile(prices, pcts)

        color = PALETTE[i % len(PALETTE)]
        for p, v in zip(pcts, pct_vals):
            fig.add_trace(go.Bar(
                x=[ticker], y=[v],
                name=f"p{p}" if i == 0 else None,
                showlegend=(i == 0),
                marker_color=f"rgba{tuple(int(color.lstrip('#')[k:k+2], 16) for k in (0, 2, 4)) + (0.6,)}",
            ))

    fig.update_layout(
        title="Simulated Terminal Price Distribution (S₀ = 100)",
        yaxis_title="Terminal Price",
        template="plotly_dark",
        barmode="overlay",
        height=450,
    )
    return fig


def per_asset_var_bar(asset_stats_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing VaR99 and CVaR99 across assets."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="VaR₉₉",
        x=asset_stats_df.index,
        y=asset_stats_df["var_99"] * 100,
        marker_color="#ffd93d",
    ))
    fig.add_trace(go.Bar(
        name="CVaR₉₉",
        x=asset_stats_df.index,
        y=asset_stats_df["cvar_99"] * 100,
        marker_color="#ff6b6b",
    ))
    fig.update_layout(
        title="Per-Asset VaR₉₉ vs CVaR₉₉",
        yaxis_title="Loss (%)",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    return fig
