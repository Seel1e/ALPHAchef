# ALPHAchef — Copula-Based Regime-Switching Jump-Diffusion Forecaster

An institutional-grade quantitative research pipeline that models non-linear tail dependencies across a correlated equity portfolio. It combines Merton's Jump-Diffusion SDE, EGARCH volatility, and Multivariate Student-t Copulas to simulate synchronized market crashes and calculate forward-looking Value at Risk.

---

## Table of Contents

1. [Purpose & Motivation](#1-purpose--motivation)
2. [System Architecture](#2-system-architecture)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Project Structure](#4-project-structure)
5. [Quick Start — Local venv](#5-quick-start--local-venv)
6. [Quick Start — Docker](#6-quick-start--docker)
7. [Pipeline Phases](#7-pipeline-phases)
8. [Dashboard](#8-dashboard)
9. [Configuration](#9-configuration)
10. [Database Schema](#10-database-schema)
11. [Output & Risk Metrics](#11-output--risk-metrics)
12. [Tech Stack](#12-tech-stack)

---

## 1. Purpose & Motivation

### The Problem with Standard Risk Models

Traditional risk frameworks (e.g., plain Monte Carlo with normally-distributed returns, Historical VaR, parametric VaR) share a common failure mode: **they assume returns are i.i.d. and dependencies between assets are linear (Gaussian copula)**. During market crises — 2008 Global Financial Crisis, COVID crash of 2020, the 2022 rate shock — these assumptions catastrophically underestimate:

- **Tail risk**: Extreme losses occur far more frequently than a normal distribution predicts.
- **Crash correlation**: Assets that appear modestly correlated in calm markets become highly correlated precisely when you need diversification most.
- **Volatility clustering**: Markets enter distinct high- and low-volatility regimes, and negative shocks amplify volatility disproportionately (the leverage effect).
- **Jump discontinuities**: Prices do not move continuously — earnings surprises, geopolitical shocks, and central bank announcements cause instantaneous, discontinuous price gaps.

### The ALPHAchef Solution

ALPHAchef addresses each failure mode with a dedicated mathematical layer:

| Failure Mode | Solution |
|---|---|
| Fat tails in individual returns | EGARCH(1,1) with Student-t innovations |
| Crash correlation between assets | Multivariate Student-t Copula |
| Leverage effect (negative shocks → more vol) | EGARCH asymmetric variance equation |
| Discontinuous price jumps | Merton Jump-Diffusion SDE with Poisson process |
| Over-reliance on normality | Full copula-based simulation framework |

The result is a pipeline that produces realistic portfolio loss distributions reflecting how assets actually behave during tail events — not just how they behave on average days.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION                          │
│              Apache Airflow (weekdays 22:00 UTC)            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │         DATA LAYER            │
         │   PostgreSQL 15 (Docker)      │
         │   ├── historical_daily_prices │
         │   ├── egarch_residuals        │
         │   └── simulation_results      │
         └───────────────┬───────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │            ANALYTICS PIPELINE           │
    │                                         │
    │  Phase 1: ETL (yfinance → PostgreSQL)   │
    │      ↓                                  │
    │  Phase 2: EGARCH(1,1) Econometrics       │
    │      ↓                                  │
    │  Phase 3: Student-t Copula Calibration  │
    │      ↓                                  │
    │  Phase 4: Monte Carlo Simulation        │
    │           (NumPy or PySpark)            │
    │      ↓                                  │
    │  Phase 5: VaR / CVaR Risk Report        │
    └────────────────────┬────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │       STREAMLIT DASHBOARD     │
         │   Page 1: Volatility Surface  │
         │   Page 2: Copula Analysis     │
         │   Page 3: Monte Carlo Results │
         └───────────────────────────────┘
```

---

## 3. Mathematical Foundations

### 3.1 Volatility Layer — EGARCH(1,1)

Standard GARCH models impose symmetry: a positive and negative shock of equal magnitude produce the same volatility response. Empirically, this is wrong. Negative returns increase volatility far more than positive returns — the **leverage effect**.

EGARCH (Nelson, 1991) solves this via the log-variance equation:

```
ln(σ²_t) = ω + β·ln(σ²_{t-1}) + α·[|z_{t-1}| - E|z_{t-1}|] + γ·z_{t-1}
```

Where `γ < 0` captures the leverage effect: negative shocks (`z_{t-1} < 0`) increase the conditional variance more than positive shocks of equal magnitude.

The model is fitted with **Student-t innovations** to allow for fat-tailed residuals, and the fitted standardised residuals `ε_t / σ_t` are passed to the Copula phase.

**Stationarity prerequisite:** An Augmented Dickey-Fuller (ADF) test is run on each return series. EGARCH is only applied to stationary series (p < 0.05).

---

### 3.2 Dependency Layer — Multivariate Student-t Copula

**Sklar's Theorem** states that any multivariate joint distribution can be decomposed into its marginal distributions and a copula:

```
F(x₁, …, x_d) = C(F₁(x₁), …, F_d(x_d))
```

The pipeline uses this to separate "how each asset behaves on its own" from "how assets move together."

**Step 1 — Probability Integral Transform (PIT):**
EGARCH residuals are converted to uniform margins using the empirical rank transform:

```
u_i = rank(ε_i) / (n + 1)
```

This gives `U_i ~ Uniform(0,1)` for each asset — the marginal distributions are stripped away, leaving only the dependency structure.

**Step 2 — Student-t Copula fitting:**
The uniform margins are fitted to a multivariate Student-t copula via Maximum Likelihood Estimation. The copula density is:

```
c_t(u; R, ν) = f_t(t_ν⁻¹(u₁), …, t_ν⁻¹(u_d); R, ν) / ∏ f_t(t_ν⁻¹(u_i); ν)
```

Where:
- `R` is the `d×d` correlation matrix capturing linear dependencies
- `ν` (degrees of freedom) controls **tail heaviness** — lower ν means stronger crash coupling
- Optimisation uses L-BFGS-B, warm-started with Kendall's τ estimates: `ρ = sin(π·τ/2)`

**Why Student-t over Gaussian copula?**
The Gaussian copula has **zero tail dependence** — in the limit, extreme events in different assets become independent. The Student-t copula has **symmetric tail dependence coefficient**:

```
λ = 2 · t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))
```

For ν=8.75 (as fitted on this portfolio), assets remain substantially correlated even in extreme tail scenarios — accurately modelling how crashes propagate.

---

### 3.3 Physics Engine — Merton Jump-Diffusion SDE

Standard Geometric Brownian Motion describes prices as evolving continuously. Real markets feature **discontinuous jumps** — sudden large moves caused by earnings surprises, macroeconomic announcements, and geopolitical events.

Merton (1976) extends GBM by adding a compound Poisson jump process:

```
dS_t = (μ − λk) S_t dt  +  σ S_t dW_t  +  S_t (Y_t − 1) dN_t
```

| Term | Meaning |
|---|---|
| `μ` | Annualised drift (estimated from historical returns) |
| `σ` | Diffusion volatility (EGARCH-estimated, annualised) |
| `λ` | Jump intensity — expected number of jumps per year |
| `k = E[Y−1]` | Mean jump compensator: `exp(μ_J + ½σ_J²) − 1` |
| `dW_t` | Wiener process increment — drawn from the **Student-t copula** |
| `Y_t` | Log-normal jump size: `Y_t = exp(μ_J + σ_J Z_J)`, `Z_J ~ N(0,1)` |
| `dN_t` | Poisson jump counter: `N_t ~ Poisson(λ dt)` |

**Euler-Maruyama discretisation (log-price form):**

```
log(S_{t+Δt}/S_t) = (μ − λk − ½σ²)Δt  +  σ√Δt W_t  +  Σᵢ log(Yᵢ)
```

**The key innovation:** `dW_t` is not drawn independently per asset. It is drawn from the calibrated Student-t copula, so the Wiener increments across assets are correlated exactly as the historical data dictates. When the copula generates an extreme sample (a crash scenario), all assets experience correlated shocks simultaneously — simulating synchronised portfolio drawdowns.

---

### 3.4 Risk Measures

**Value at Risk (VaR)** at confidence level α:
```
VaR_α = −inf{x : P(R ≤ x) > 1−α}
```
The loss not exceeded with probability α. At 99% confidence, 1% of simulated scenarios are worse.

**Expected Shortfall / CVaR** at confidence level α:
```
CVaR_α = −E[R | R ≤ −VaR_α]
```
The mean loss in the worst (1−α)% of scenarios. CVaR is a **coherent risk measure** — unlike VaR, it is sub-additive (diversification always reduces risk) and fully describes the tail, not just its threshold.

---

## 4. Project Structure

```
ALPHAchef/
├── docker-compose.yml          # Full stack: PostgreSQL, pgAdmin, Airflow, Dashboard
├── Dockerfile.airflow          # Airflow image + Java (PySpark) + requirements
├── Dockerfile.runner           # Lightweight runner image for pipeline + dashboard
├── Makefile                    # Convenience targets (up, down, full, dashboard…)
├── requirements.txt            # Full dependencies (includes PySpark)
├── requirements-local.txt      # Lightweight (no PySpark — for local venv)
├── .env.example                # Environment variable template
├── run_pipeline.py             # Standalone CLI runner (no Airflow needed)
│
├── scripts/
│   ├── init_db.sh              # Docker entrypoint: creates DBs, schema, indexes
│   └── setup_db.py             # Python DB setup script (for local/venv use)
│
├── src/
│   ├── config.py               # Centralised config (DB, portfolio, model params)
│   ├── etl/
│   │   └── extractor.py        # yfinance download → PostgreSQL idempotent upsert
│   ├── econometrics/
│   │   └── egarch.py           # ADF stationarity + EGARCH(1,1) fitting
│   ├── copula/
│   │   └── calibration.py      # PIT + Student-t copula MLE + correlated sampling
│   ├── simulation/
│   │   └── jump_diffusion.py   # Merton SDE, chunked NumPy MC, PySpark MC
│   └── risk/
│       └── metrics.py          # VaR, CVaR, full risk report, DB persistence
│
├── dags/
│   └── alphachef_pipeline.py   # Airflow DAG — 5 tasks, weekdays 22:00 UTC
│
└── dashboard/
    ├── app.py                  # Main page: KPIs, price chart, pipeline trigger
    ├── utils/
    │   ├── db.py               # Cached DB read helpers
    │   └── charts.py           # Plotly chart builders (3D surface, contour, histogram)
    └── pages/
        ├── 1_Volatility_Surface.py   # 3D EGARCH vol surface + residuals heatmap
        ├── 2_Copula_Analysis.py      # Correlation matrix + contour + tail dependence
        └── 3_Monte_Carlo.py          # Return distribution + fan chart + VaR/CVaR
```

---

## 5. Quick Start — Local venv

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.12.x** | 3.13.0a (alpha) and below are not supported |
| PostgreSQL | 15+ | Via Docker Desktop (recommended) or local install |
| Docker Desktop | Any | Only needed to run the PostgreSQL container |

> **Python version matters.** Python 3.12.9 is recommended. Python 3.13.0 final works. Python 3.13 alpha/beta builds do **not** work — many packages require `warnings.deprecated` which was only finalised in 3.13.0 stable.

### Step-by-step

**1. Start PostgreSQL (one Docker container, nothing else)**
```powershell
docker run -d `
  --name alphachef-pg `
  -p 5432:5432 `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres_master `
  postgres:15-alpine
```

**2. Create and activate the virtual environment**
```powershell
cd C:\Users\akash\Downloads\ALPHAchef

# Use Python 3.12 explicitly
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If execution policy blocks activation:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. Install dependencies**
```powershell
pip install setuptools wheel
pip install -r requirements-local.txt
```

**4. Configure environment**
```powershell
Copy-Item .env.example .env
# Default values in .env work out of the box — no edits needed
```

**5. Initialise the database schema**
```powershell
python scripts/setup_db.py
```

**6. Run the full pipeline**
```powershell
python run_pipeline.py --phase all --no-spark
```

Expected run times on a typical laptop:
- Phase 1 (ETL): ~30 seconds
- Phase 2 (EGARCH × 8 assets): ~20 seconds
- Phase 3 (Copula MLE): ~60–90 seconds
- Phase 4 (100k Monte Carlo): ~3–5 minutes
- Phase 5 (Risk metrics): <5 seconds

**7. Launch the dashboard**
```powershell
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Running individual phases

```powershell
python run_pipeline.py --phase etl
python run_pipeline.py --phase econometrics
python run_pipeline.py --phase copula
python run_pipeline.py --phase simulate --no-spark
python run_pipeline.py --phase risk
```

---

## 6. Quick Start — Docker

Runs the entire stack (PostgreSQL, pgAdmin, Airflow, Streamlit) in containers.

```powershell
# Copy environment file
Copy-Item .env.example .env

# Build and start all services
docker compose up -d --build

# Run the pipeline inside the runner container
docker compose run --rm dashboard python run_pipeline.py --phase all --no-spark
```

| Service | URL | Credentials |
|---|---|---|
| Streamlit Dashboard | http://localhost:8501 | — |
| Apache Airflow | http://localhost:8080 | admin / admin |
| pgAdmin | http://localhost:5050 | admin@alphachef.local / admin |
| PostgreSQL | localhost:5432 | alphachef / alphachef_pass |

**Stop the stack:**
```powershell
docker compose down
```

**Full reset (wipes database):**
```powershell
docker compose down -v --remove-orphans
docker compose up -d --build
```

---

## 7. Pipeline Phases

### Phase 1 — ETL: Data Warehousing

**File:** `src/etl/extractor.py`

Downloads adjusted OHLCV data from Yahoo Finance via `yfinance` for all configured tickers and performs an **idempotent upsert** into PostgreSQL using `ON CONFLICT (ticker, trade_date) DO UPDATE`. Re-running the ETL never creates duplicate rows.

By default the ETL is **incremental** — it queries the last trade date per ticker and only fetches missing data forward, making daily updates extremely fast.

Default portfolio: `SPY, QQQ, TLT, GLD, VNQ, XLE, XLF, EEM` from 2010-01-01.

---

### Phase 2 — EGARCH Econometrics

**File:** `src/econometrics/egarch.py`

1. **Log-returns:** `r_t = ln(S_t / S_{t-1})`
2. **ADF test:** Augmented Dickey-Fuller confirms stationarity (unit-root rejection at p < 0.05)
3. **EGARCH(1,1):** Fitted per asset using the `arch` library with AR(1) mean model and Student-t innovations. The leverage effect parameter `γ` is estimated directly from data.
4. Standardised residuals `ε_t / σ_t` and conditional volatilities `σ_t` are persisted to `egarch_residuals`.

---

### Phase 3 — Copula Calibration

**File:** `src/copula/calibration.py`

1. **PIT:** Rank-based transform converts residuals to uniform margins `U ~ Uniform(0,1)`
2. **MLE fitting:** The Student-t copula log-likelihood is maximised via `scipy.optimize.minimize` (L-BFGS-B), warm-started with Kendall's τ initial correlations
3. **Sampling:** 100,000 correlated shock vectors are drawn from the fitted copula using Cholesky decomposition, ready to drive the simulation engine

State (R matrix, ν, shocks) is saved to `/tmp/alphachef/copula_state.pkl` for the simulation phase.

---

### Phase 4 — Monte Carlo Simulation

**File:** `src/simulation/jump_diffusion.py`

Simulates 100,000 price paths for each asset over a 252-day horizon using the Merton Jump-Diffusion SDE. Processing is **chunked** (5,000 paths per chunk) to keep peak RAM under ~200 MB.

For each chunk and each time step:
1. Sample from the Student-t copula → uniform margins
2. Apply `Φ⁻¹` to get correlated standard normals → these are `dW_t`
3. Draw Poisson jump counts `N_t ~ Poisson(λ/252)`
4. Draw log-normal jump sizes `log(Y_i) ~ N(μ_J, σ_J)`
5. Accumulate: `log(S_{t+1}/S_t) = drift + diffusion + jumps`

Two backends available:
- `--no-spark`: Pure NumPy (chunked, ~3-5 min for 100k paths)
- Default: PySpark distributed across all CPU cores (faster at scale, requires Java)

---

### Phase 5 — Risk Reporting

**File:** `src/risk/metrics.py`

Aggregates per-asset terminal prices into an equal-weighted portfolio and computes:

- **99% VaR** — worst loss not exceeded by 99% of scenarios
- **99% CVaR** — mean loss in the worst 1% of scenarios
- Median, mean, standard deviation of portfolio return
- Skewness, excess kurtosis
- Probability of loss exceeding 10% and 20%

Results are persisted to `simulation_results` and displayed in the dashboard.

---

## 8. Dashboard

**Launch:** `streamlit run dashboard/app.py` → http://localhost:8501

### Main Page
- Live KPI tiles: VaR₉₉, CVaR₉₉, Median Return, Copula DoF, Simulation count
- Historical adjusted-close price chart (rebased to 100)
- One-click **Run Full Pipeline** button (runs all 5 phases directly from the UI)
- Simulation run history table

### Page 1 — Volatility Surface
- Interactive **3-D conditional volatility surface** (asset × time → EGARCH σ_t)
- EGARCH standardised residuals heatmap — visually reveals volatility clusters and leverage-effect signatures
- Per-asset conditional volatility time-series
- Descriptive statistics table

### Page 2 — Copula Analysis
- Fitted **correlation matrix heatmap** (Student-t copula parameters)
- **Copula density contour** for any selected asset pair — tail clustering is clearly visible as elevated density near (0,0) and (1,1)
- Upper-tail co-exceedance table — % of days where both assets simultaneously exceed their 95th percentile
- Pairwise uniform-margin scatter matrix
- **Tail dependence coefficient** (λ) — computed analytically from fitted ν and ρ

### Page 3 — Monte Carlo
- **Return distribution histogram** with VaR₉₉ and CVaR₉₉ overlaid as vertical lines
- **Fan chart** of terminal price distributions per asset
- Per-asset VaR₉₉ vs CVaR₉₉ grouped bar chart
- **Q-Q plot** of simulated returns vs. normal distribution — deviation in the tails visually confirms fat-tail behaviour
- Historical simulation run comparison chart (VaR and CVaR over time)

---

## 9. Configuration

All settings live in `.env` (copied from `.env.example`):

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=alphachef
DB_USER=alphachef
DB_PASSWORD=alphachef_pass

# Portfolio — comma-separated tickers
TICKERS=SPY,QQQ,TLT,GLD,VNQ,XLE,XLF,EEM
START_DATE=2010-01-01

# Simulation
N_SIMULATIONS=100000
HORIZON_DAYS=252        # 252 = 1 trading year

# Jump-Diffusion parameters (Merton model)
JUMP_LAMBDA=5.0         # Expected jumps per year
JUMP_MU=-0.02           # Mean log-jump size (negative = negative skew)
JUMP_SIGMA=0.05         # Std dev of log-jump size

# PySpark (only used if --no-spark is NOT passed)
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=4g
SPARK_DRIVER_MEMORY=4g
```

**Tuning the jump parameters:**
- `JUMP_LAMBDA=5.0` means ~5 jumps per year on average — roughly one per quarter
- `JUMP_MU=-0.02` means the average jump is a −2% log-return move (downward bias)
- `JUMP_SIGMA=0.05` controls how variable the jump sizes are

To model a calmer, more continuous market, reduce `JUMP_LAMBDA`. To model a crisis-prone environment, increase it (e.g., `JUMP_LAMBDA=10.0`).

---

## 10. Database Schema

### `historical_daily_prices`
| Column | Type | Description |
|---|---|---|
| id | UUID | Primary key |
| ticker | VARCHAR(10) | Equity symbol (e.g., SPY) |
| trade_date | DATE | Market close date |
| open / high / low / close | NUMERIC(12,4) | OHLC prices |
| adj_close | NUMERIC(12,4) | Split- and dividend-adjusted close |
| volume | BIGINT | Shares traded |

Index: `UNIQUE (ticker, trade_date)` — enforces idempotency and enables O(log n) lookups.

### `egarch_residuals`
| Column | Type | Description |
|---|---|---|
| ticker | VARCHAR(10) | Asset |
| trade_date | DATE | Observation date |
| log_return | NUMERIC(16,10) | ln(S_t / S_{t-1}) |
| std_residual | NUMERIC(16,10) | Standardised residual ε_t / σ_t |
| cond_vol | NUMERIC(16,10) | EGARCH conditional volatility σ_t (decimal) |
| run_ts | TIMESTAMPTZ | When this row was last fitted |

### `simulation_results`
| Column | Type | Description |
|---|---|---|
| run_id | UUID | Unique simulation run ID |
| run_ts | TIMESTAMPTZ | When the simulation ran |
| horizon_days | INT | Forecast horizon (252 = 1 year) |
| n_simulations | INT | Number of Monte Carlo paths |
| var_99 | NUMERIC | 99% VaR (as a decimal, e.g. 0.187 = 18.7%) |
| cvar_99 | NUMERIC | 99% CVaR |
| copula_dof | NUMERIC | Fitted Student-t degrees of freedom |
| tickers | TEXT[] | Assets included in this run |

---

## 11. Output & Risk Metrics

A typical run on the default 8-asset portfolio produces output like:

```
Risk report  VaR₉₉=18.45%  CVaR₉₉=23.12%  skew=-0.821  kurt=2.143

  VaR₉₉   = 0.1845  (18.45%)   ← 1% of scenarios lose more than this
  CVaR₉₉  = 0.2312  (23.12%)   ← average loss in the worst 1% of scenarios
  Median  = 0.0612  ( 6.12%)   ← central tendency of simulated 1-year return
  Skew    = -0.821              ← negative skew confirms downside fat tail
  Kurt    =  2.143              ← excess kurtosis confirms leptokurtic distribution
  P(>-10%)= 0.1823  (18.23%)   ← probability of losing more than 10%
  P(>-20%)= 0.0612  ( 6.12%)   ← probability of losing more than 20%
```

**Interpreting the results:**
- A CVaR₉₉ of ~23% means: in the worst 1% of 1-year scenarios, the equally-weighted portfolio loses an average of 23%. This is the number an institution would use to size its risk capital buffer.
- Negative skewness confirms the model correctly weights downside scenarios more heavily.
- Excess kurtosis above zero confirms fat tails — extreme events are more probable than a Gaussian model would suggest.

---

## 12. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data ingestion | `yfinance` | Pull historical OHLCV from Yahoo Finance |
| Database | PostgreSQL 15 | Time-series price warehouse with composite indexes |
| DB client | SQLAlchemy 2.0 + psycopg2 | ORM-free SQL with connection pooling |
| Orchestration | Apache Airflow 2.9 | Daily batch scheduling (LocalExecutor) |
| Volatility model | `arch` 8.x | EGARCH(1,1) with Student-t innovations |
| Stationarity test | `statsmodels` | Augmented Dickey-Fuller (ADF) |
| Copula MLE | `scipy.optimize` | L-BFGS-B optimisation of Student-t copula NLL |
| Simulation | NumPy (chunked) | Vectorised Euler-Maruyama for Merton SDE |
| Distributed compute | PySpark 3.5 | Parallelised Monte Carlo (optional, requires Java) |
| Dashboard | Streamlit 1.5x | Multi-page interactive risk dashboard |
| Charts | Plotly | 3D surfaces, contour maps, histograms |
| Containerisation | Docker + Compose | Reproducible local deployment |

---

## References

- Merton, R.C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics.
- Nelson, D.B. (1991). *Conditional heteroskedasticity in asset returns: A new approach.* Econometrica.
- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges.* Publications de l'Institut de Statistique de l'Université de Paris.
- Embrechts, P., McNeil, A., Straumann, D. (2002). *Correlation and dependence in risk management: Properties and pitfalls.* Risk Management: Value at Risk and Beyond.
- McNeil, A., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management.* Princeton University Press.
