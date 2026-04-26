# ALPHAchef
**Copula-Based Regime-Switching Jump-Diffusion Forecaster**

Institutional-grade quantitative risk pipeline modelling non-linear dependencies and tail-risk across a correlated equity portfolio.

---

## Stack
| Layer | Tech |
|---|---|
| Storage | PostgreSQL |
| Orchestration | Apache Airflow |
| ETL | Python · yfinance · SQLAlchemy |
| Econometrics | arch · statsmodels · scipy |
| Simulation | PySpark · NumPy |
| Dashboard | Streamlit · Plotly |

---

## Math

**Volatility** — EGARCH(1,1) captures the leverage effect (negative shocks amplify volatility more than positive ones).

**Dependency** — Multivariate Student-t Copula via Sklar's Theorem maps how assets crash together in tail events.

**Simulation** — Merton Jump-Diffusion SDE:

$$dS_t = (\mu - \lambda k)\, S_t\, dt + \sigma S_t\, dW_t + S_t(Y_t - 1)\, dN_t$$

---

## Pipeline

| Phase | Description |
|---|---|
| 1 | Infrastructure — PostgreSQL schema, composite indexes, idempotent ETL upserts |
| 2 | Econometrics — log returns, ADF stationarity tests, EGARCH(1,1) residuals |
| 3 | Copula — PIT to uniform margins, Student-t copula MLE fit |
| 4 | Simulation — 100,000 Monte Carlo paths with copula-correlated shocks |
| 5 | Risk — 99% VaR, CVaR, Streamlit dashboard with 3D vol surface & copula maps |

---

## Quick Start

```bash
cp .env.example .env
python scripts/setup_db.py
python run_pipeline.py --phase all --no-spark
streamlit run dashboard/app.py
```
