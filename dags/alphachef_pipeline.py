"""
ALPHAchef — Airflow DAG

Runs the 5-phase pipeline daily at 22:00 UTC (after US market close).

Phase 1  etl_extraction      Pull OHLCV from yfinance, upsert to PostgreSQL
Phase 2  egarch_econometrics  ADF tests + EGARCH(1,1) residual extraction
Phase 3  copula_calibration   PIT + multivariate Student-t copula fit
Phase 4  monte_carlo          100k Merton Jump-Diffusion paths via PySpark
Phase 5  risk_reporting       VaR, CVaR, persist results
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/alphachef")

_DEFAULT_ARGS = {
    "owner": "alphachef",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}


# ── Task callables ─────────────────────────────────────────────────────────────

def _phase1_etl(**ctx):
    from src.etl.extractor import run_etl
    summary = run_etl(incremental=True)
    failed = [t for t, r in summary.items() if r["status"] != "ok"]
    if failed:
        raise RuntimeError(f"ETL failed for: {failed}")
    total_rows = sum(r["rows"] for r in summary.values())
    print(f"ETL complete — {total_rows} total rows upserted across {len(summary)} tickers")
    return total_rows


def _phase2_econometrics(**ctx):
    from src.econometrics.egarch import run_econometrics
    residuals, log_returns = run_econometrics()
    n_tickers = len(residuals)
    print(f"EGARCH complete — {n_tickers} assets processed")
    return n_tickers


def _phase3_copula(**ctx):
    import pickle
    from src.econometrics.egarch import run_econometrics
    from src.copula.calibration import (
        fit_student_t_copula,
        probability_integral_transform,
        sample_copula,
    )
    from src.config import MODEL

    residuals, _ = run_econometrics()
    uniform_margins = probability_integral_transform(residuals)
    R, nu = fit_student_t_copula(uniform_margins)
    shocks = sample_copula(R, nu, MODEL.n_simulations)

    os.makedirs("/tmp/alphachef", exist_ok=True)
    state = {"R": R, "nu": nu, "shocks": shocks, "tickers": list(residuals.keys())}
    with open("/tmp/alphachef/copula_state.pkl", "wb") as fh:
        pickle.dump(state, fh, protocol=5)

    print(f"Copula fitted: dof={nu:.2f}, corr_shape={R.shape}")
    return float(nu)


def _phase4_simulation(**ctx):
    import pickle
    from src.econometrics.egarch import run_econometrics
    from src.simulation.jump_diffusion import estimate_params, run_simulation

    with open("/tmp/alphachef/copula_state.pkl", "rb") as fh:
        state = pickle.load(fh)

    _, log_returns = run_econometrics()
    params = estimate_params(log_returns)

    terminal = run_simulation(
        params,
        state["R"],
        state["nu"],
        tickers=state["tickers"],
        use_spark=True,
    )

    with open("/tmp/alphachef/simulation_results.pkl", "wb") as fh:
        pickle.dump({"terminal": terminal, "nu": state["nu"]}, fh, protocol=5)

    print(f"Simulation complete — {len(terminal)} assets, "
          f"{len(next(iter(terminal.values())))} paths each")


def _phase5_risk(**ctx):
    import pickle
    from src.etl.extractor import get_engine
    from src.risk.metrics import persist_report, risk_report
    from src.config import PORTFOLIO

    with open("/tmp/alphachef/simulation_results.pkl", "rb") as fh:
        state = pickle.load(fh)

    report = risk_report(
        state["terminal"],
        copula_dof=state["nu"],
        tickers=PORTFOLIO.tickers,
    )
    engine = get_engine()
    run_id = persist_report(report, engine)

    print(
        f"Risk report persisted [run_id={run_id}]  "
        f"VaR99={report['var_99']:.2%}  CVaR99={report['cvar_99']:.2%}"
    )
    return run_id


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="alphachef_daily_pipeline",
    default_args=_DEFAULT_ARGS,
    description="ALPHAchef: ETL → EGARCH → Copula → Monte Carlo → Risk",
    schedule="0 22 * * 1-5",    # weekdays at 22:00 UTC (after NYSE close)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["alphachef", "quant", "risk"],
) as dag:

    t1 = PythonOperator(task_id="etl_extraction",      python_callable=_phase1_etl)
    t2 = PythonOperator(task_id="egarch_econometrics", python_callable=_phase2_econometrics)
    t3 = PythonOperator(task_id="copula_calibration",  python_callable=_phase3_copula)
    t4 = PythonOperator(task_id="monte_carlo",         python_callable=_phase4_simulation)
    t5 = PythonOperator(task_id="risk_reporting",      python_callable=_phase5_risk)

    t1 >> t2 >> t3 >> t4 >> t5
