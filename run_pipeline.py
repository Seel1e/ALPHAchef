"""
ALPHAchef — Standalone CLI runner.

Executes the full 5-phase pipeline without requiring Airflow.
Useful for local development, first-run setup, and CI.

Usage
-----
    python run_pipeline.py --phase all
    python run_pipeline.py --phase etl
    python run_pipeline.py --phase econometrics
    python run_pipeline.py --phase copula
    python run_pipeline.py --phase simulate --no-spark
    python run_pipeline.py --phase risk
"""

import argparse
import logging
import os
import pickle
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("alphachef.runner")

STATE_DIR = "/tmp/alphachef"
os.makedirs(STATE_DIR, exist_ok=True)


# ── Phase functions ────────────────────────────────────────────────────────────

def phase_etl():
    from src.etl.extractor import run_etl
    logger.info("═" * 60)
    logger.info("PHASE 1 — ETL  (incremental yfinance → PostgreSQL)")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    summary = run_etl(incremental=True)
    elapsed = time.perf_counter() - t0
    ok = [t for t, r in summary.items() if r["status"] == "ok"]
    err = [t for t, r in summary.items() if r["status"] != "ok"]
    total_rows = sum(r["rows"] for r in summary.values())
    logger.info("ETL done in %.1fs  |  rows=%d  |  ok=%s  |  errors=%s",
                elapsed, total_rows, ok, err)
    if err:
        logger.error("ETL failed for: %s", err)


def phase_econometrics():
    from src.econometrics.egarch import run_econometrics
    logger.info("═" * 60)
    logger.info("PHASE 2 — EGARCH econometrics (ADF + EGARCH(1,1))")
    logger.info("═" * 60)
    t0 = time.perf_counter()
    residuals, log_returns = run_econometrics()
    elapsed = time.perf_counter() - t0
    logger.info("Econometrics done in %.1fs  |  assets=%d", elapsed, len(residuals))

    _save_state("econometrics_state.pkl", {
        "residuals": residuals,
        "log_returns": log_returns,
    })


def phase_copula():
    from src.copula.calibration import (
        fit_student_t_copula,
        probability_integral_transform,
        sample_copula,
    )
    from src.config import MODEL

    logger.info("═" * 60)
    logger.info("PHASE 3 — Copula calibration (PIT + Student-t MLE)")
    logger.info("═" * 60)

    state = _load_state("econometrics_state.pkl")
    if state is None:
        logger.error("econometrics_state.pkl not found — run phase 2 first.")
        sys.exit(1)

    t0 = time.perf_counter()
    uniforms = probability_integral_transform(state["residuals"])
    R, nu = fit_student_t_copula(uniforms)
    shocks = sample_copula(R, nu, MODEL.n_simulations)
    elapsed = time.perf_counter() - t0

    logger.info("Copula done in %.1fs  |  dof=%.2f", elapsed, nu)
    _save_state("copula_state.pkl", {
        "R": R, "nu": nu, "shocks": shocks,
        "tickers": list(state["residuals"].keys()),
        "uniforms": uniforms,
    })


def phase_simulate(use_spark: bool = True):
    from src.simulation.jump_diffusion import estimate_params, run_simulation

    logger.info("═" * 60)
    logger.info("PHASE 4 — Monte Carlo (Merton Jump-Diffusion, %s)",
                "PySpark" if use_spark else "NumPy")
    logger.info("═" * 60)

    econ = _load_state("econometrics_state.pkl")
    cop  = _load_state("copula_state.pkl")
    if not econ or not cop:
        logger.error("Missing upstream state files — run phases 2 & 3 first.")
        sys.exit(1)

    t0 = time.perf_counter()
    params = estimate_params(econ["log_returns"])
    terminal = run_simulation(params, cop["R"], cop["nu"],
                              tickers=cop["tickers"], use_spark=use_spark)
    elapsed = time.perf_counter() - t0

    n_paths = len(next(iter(terminal.values())))
    logger.info("Simulation done in %.1fs  |  paths=%d  |  assets=%d",
                elapsed, n_paths, len(terminal))

    _save_state("simulation_state.pkl", {
        "terminal": terminal,
        "nu": cop["nu"],
        "params": params,
    })


def phase_risk():
    from src.etl.extractor import get_engine
    from src.risk.metrics import persist_report, risk_report
    from src.config import PORTFOLIO

    logger.info("═" * 60)
    logger.info("PHASE 5 — Risk metrics (VaR / CVaR)")
    logger.info("═" * 60)

    state = _load_state("simulation_state.pkl")
    if state is None:
        logger.error("simulation_state.pkl not found — run phase 4 first.")
        sys.exit(1)

    t0 = time.perf_counter()
    report = risk_report(
        state["terminal"],
        copula_dof=state["nu"],
        tickers=PORTFOLIO.tickers,
    )
    engine = get_engine()
    run_id = persist_report(report, engine)
    elapsed = time.perf_counter() - t0

    logger.info("Risk report saved in %.1fs  |  run_id=%s", elapsed, run_id)
    logger.info("  VaR₉₉   = %.4f  (%.2f%%)", report["var_99"], report["var_99"] * 100)
    logger.info("  CVaR₉₉  = %.4f  (%.2f%%)", report["cvar_99"], report["cvar_99"] * 100)
    logger.info("  Median  = %.4f  (%.2f%%)", report["median_return"], report["median_return"] * 100)
    logger.info("  Skew    = %.4f", report["skewness"])
    logger.info("  Kurt    = %.4f", report["kurtosis"])
    logger.info("  P(>-10%%)= %.4f", report["pct_loss_gt10"])
    logger.info("  P(>-20%%)= %.4f", report["pct_loss_gt20"])


# ── State helpers ──────────────────────────────────────────────────────────────

def _save_state(filename: str, obj) -> None:
    path = os.path.join(STATE_DIR, filename)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    logger.debug("State saved → %s", path)


def _load_state(filename: str):
    path = os.path.join(STATE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALPHAchef pipeline runner")
    parser.add_argument(
        "--phase",
        choices=["all", "etl", "econometrics", "copula", "simulate", "risk"],
        default="all",
    )
    parser.add_argument("--no-spark", action="store_true",
                        help="Use NumPy instead of PySpark for simulation")
    args = parser.parse_args()

    use_spark = not args.no_spark

    if args.phase == "all":
        t_total = time.perf_counter()
        phase_etl()
        phase_econometrics()
        phase_copula()
        phase_simulate(use_spark)
        phase_risk()
        logger.info("Full pipeline completed in %.1fs", time.perf_counter() - t_total)

    elif args.phase == "etl":
        phase_etl()
    elif args.phase == "econometrics":
        phase_econometrics()
    elif args.phase == "copula":
        phase_copula()
    elif args.phase == "simulate":
        phase_simulate(use_spark)
    elif args.phase == "risk":
        phase_risk()


if __name__ == "__main__":
    main()
