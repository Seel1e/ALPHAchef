import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "alphachef")
    user: str = os.getenv("DB_USER", "alphachef")
    password: str = os.getenv("DB_PASSWORD", "alphachef_pass")

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass
class PortfolioConfig:
    tickers: List[str] = field(
        default_factory=lambda: os.getenv(
            "TICKERS", "SPY,QQQ,TLT,GLD,VNQ,XLE,XLF,EEM"
        ).split(",")
    )
    start_date: str = os.getenv("START_DATE", "2010-01-01")


@dataclass
class ModelConfig:
    # EGARCH(p, q)
    egarch_p: int = 1
    egarch_q: int = 1

    # Jump-Diffusion (Merton)
    n_simulations: int = int(os.getenv("N_SIMULATIONS", "100000"))
    horizon_days: int = int(os.getenv("HORIZON_DAYS", "252"))
    jump_lambda: float = float(os.getenv("JUMP_LAMBDA", "5.0"))   # avg jumps / year
    jump_mu: float = float(os.getenv("JUMP_MU", "-0.02"))          # mean log-jump
    jump_sigma: float = float(os.getenv("JUMP_SIGMA", "0.05"))     # std  log-jump

    # Risk
    var_confidence: float = 0.99


@dataclass
class SparkConfig:
    master: str = os.getenv("SPARK_MASTER", "local[*]")
    app_name: str = "ALPHAchef_MonteCarlo"
    executor_memory: str = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
    driver_memory: str = os.getenv("SPARK_DRIVER_MEMORY", "4g")


DB = DatabaseConfig()
PORTFOLIO = PortfolioConfig()
MODEL = ModelConfig()
SPARK = SparkConfig()
