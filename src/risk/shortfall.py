"""
Risk Layer — shortfall.py

Computes the Shortfall Risk Estimate (SRE) — an analogue to financial VaR.

Definition:
  SRE(α, w) = α-th percentile of forecast errors over trailing w-day window.
  P(actual < forecast − SRE) ≈ α

For α = 0.05, w = 90:
  The SRE gives a threshold such that actual demand falls below
  (forecast − SRE) roughly 5% of the time → 95% demand coverage.

Scientific Rationale:
  This mirrors Historical Simulation Value-at-Risk in finance but applied
  to inventory planning. The Stockout Risk Index (SRI) is a composite
  of SRE, CV, and Isolation Forest score, normalised to [0, 1].
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/risk.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DB_URL = (
    f"postgresql+psycopg2://"
    f"{os.environ.get('DB_USER', 'forecast_user')}:"
    f"{os.environ.get('DB_PASS', 'forecast_pass')}@"
    f"{os.environ.get('DB_HOST', 'localhost')}:"
    f"{os.environ.get('DB_PORT', '5432')}/"
    f"{os.environ.get('DB_NAME', 'demand_forecasting')}"
)

PROCESSED_DIR = Path("data/processed")

ALPHA         = 0.05    # 5th percentile → 95% coverage
WINDOW        = 90      # trailing window in days


def compute_forecast_errors(df: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Merge actuals with forecasts (XGBoost, primary model) and compute
    signed forecast errors: error = actual − forecast
    """
    log.info("Merging actuals with XGBoost forecasts for SRE ...")
    merged = df.merge(
        forecasts[["item_id", "store_id", "date", "forecast"]],
        on=["item_id", "store_id", "date"],
        how="inner",
    )
    merged["error"] = merged["sales"] - merged["forecast"]
    return merged


def compute_sre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SRE as rolling α-percentile of forecast errors.
    A negative SRE indicates a systematic bias toward overforecasting.
    """
    log.info("Computing Shortfall Risk Estimate (α=%.2f, window=%d days) ...",
             ALPHA, WINDOW)
    df = df.sort_values(["item_id", "store_id", "date"])

    df["sre"] = df.groupby(["item_id", "store_id"])["error"].transform(
        lambda x: x.rolling(WINDOW, min_periods=WINDOW // 3).quantile(ALPHA)
    )
    # Safety stock level: forecast - SRE (positive → buffer needed)
    df["safety_stock"] = (-df["sre"]).clip(lower=0)
    return df


def coverage_backtest(df: pd.DataFrame) -> dict:
    """
    Verify that actual demand falls below (forecast + lower_CI) roughly α% of time.
    Actual shortfall rate should be close to ALPHA.
    """
    log.info("Running coverage backtest ...")
    df["shortfall_breach"] = df["sales"] < (df["forecast"] + df["sre"])
    actual_rate = df["shortfall_breach"].mean()
    log.info("  Actual shortfall rate: %.4f  (target: %.4f)", actual_rate, ALPHA)
    log.info("  Calibration error: %.4f", abs(actual_rate - ALPHA))
    return {
        "actual_shortfall_rate": float(actual_rate),
        "target_alpha":          ALPHA,
        "calibration_error":     float(abs(actual_rate - ALPHA)),
    }


def compute_stockout_risk_index(df: pd.DataFrame,
                                 if_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Composite Stockout Risk Index (SRI) ∈ [0, 1]:
      SRI = 0.4 × norm(SRE_magnitude) + 0.4 × norm(CV) + 0.2 × norm(IF_score_inverted)

    Higher SRI = greater stockout risk.
    """
    log.info("Computing Stockout Risk Index (SRI) ...")

    def normalise(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-8)

    df["sre_norm"] = normalise(df["safety_stock"].fillna(0))
    df["cv_norm"]  = normalise(df.get("cv", pd.Series(0, index=df.index)).fillna(0))

    if if_scores is not None and "if_score" in if_scores.columns:
        df = df.merge(if_scores[["item_id", "store_id", "date", "if_score"]],
                      on=["item_id", "store_id", "date"], how="left")
        df["if_norm"] = normalise(-df["if_score"].fillna(0))  # invert: lower score = more anomalous
    else:
        df["if_norm"] = 0.0

    df["sri"] = 0.4 * df["sre_norm"] + 0.4 * df["cv_norm"] + 0.2 * df["if_norm"]
    log.info("SRI summary: mean=%.4f  max=%.4f", df["sri"].mean(), df["sri"].max())
    return df


def main() -> None:
    log.info("=" * 60)
    log.info("RISK: Shortfall (SRE / VaR analogue)  |  α=%.2f  window=%d",
             ALPHA, WINDOW)
    log.info("=" * 60)

    df = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    df["date"] = pd.to_datetime(df["date"])

    xgb_fc_path = PROCESSED_DIR / "xgb_test_forecasts.parquet"
    if not xgb_fc_path.exists():
        log.error("XGBoost forecasts not found — run xgboost_model.py first")
        return

    forecasts = pd.read_parquet(xgb_fc_path)
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    df = compute_forecast_errors(df, forecasts)
    df = compute_sre(df)
    stats = coverage_backtest(df)
    log.info("Coverage stats: %s", stats)

    # Attach CV from volatility if available
    vol_path = PROCESSED_DIR / "volatility.parquet"
    if vol_path.exists():
        vol = pd.read_parquet(vol_path)[["item_id", "store_id", "date", "cv"]]
        vol["date"] = pd.to_datetime(vol["date"])
        df = df.merge(vol, on=["item_id", "store_id", "date"], how="left")

    # Attach IF scores if available
    if_path = PROCESSED_DIR / "anomalies.parquet"
    if_scores = None
    if if_path.exists():
        if_scores = pd.read_parquet(if_path)[["item_id", "store_id", "date", "if_score"]]
        if_scores["date"] = pd.to_datetime(if_scores["date"])

    df = compute_stockout_risk_index(df, if_scores)

    out = df[["item_id", "store_id", "date", "sales", "forecast",
              "error", "sre", "safety_stock", "sri",
              "shortfall_breach"]]
    out.to_parquet(PROCESSED_DIR / "shortfall_risk.parquet",
                  index=False, compression="snappy")
    log.info("Shortfall risk data saved.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
