"""
Risk Layer — volatility.py

Computes rolling demand volatility metrics per item-store series.

Metrics:
  - Rolling σ (standard deviation) over 7, 14, 28-day windows
  - Coefficient of Variation (CV = σ/μ): scale-free, comparable across SKUs
  - Volatility regime classification: Low / Medium / High
  - Inter-quartile range (IQR) of daily demand as a robust spread measure

Scientific Rationale:
  CV is preferred over raw σ because demand scale differs by orders of
  magnitude across M5 items. A CV > 0.7 typically signals chronic supply
  uncertainty for that item-store pair.
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

CV_THRESHOLDS = {"low": 0.3, "medium": 0.7}   # boundaries for regime classification
ROLL_WINDOWS  = [7, 14, 28]


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility metrics per item-store series."""
    log.info("Computing rolling volatility (windows: %s) ...", ROLL_WINDOWS)
    df = df.sort_values(["item_id", "store_id", "date"])
    grp = df.groupby(["item_id", "store_id"])

    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = grp["sales"].transform(
            lambda x: x.rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"roll_std_{w}"]  = grp["sales"].transform(
            lambda x: x.rolling(w, min_periods=max(1, w // 2)).std()
        ).fillna(0)
        df[f"cv_{w}"] = (
            df[f"roll_std_{w}"] / df[f"roll_mean_{w}"].clip(lower=1e-6)
        )

    # Primary CV on 28-day window (main risk metric)
    df["cv"]         = df["cv_28"]
    df["roll_mean"]  = df["roll_mean_28"]
    df["roll_std"]   = df["roll_std_28"]

    # IQR per series (using expanding window for stability)
    df["iqr_28"] = grp["sales"].transform(
        lambda x: x.rolling(28, min_periods=14).apply(
            lambda v: np.percentile(v, 75) - np.percentile(v, 25)
        )
    )

    # Volatility regime
    def classify(cv: float) -> str:
        if cv <= CV_THRESHOLDS["low"]:
            return "Low"
        elif cv <= CV_THRESHOLDS["medium"]:
            return "Medium"
        return "High"

    df["volatility_regime"] = df["cv"].apply(classify)

    log.info("Volatility regime distribution:")
    log.info(df["volatility_regime"].value_counts().to_string())
    return df


def main() -> None:
    log.info("=" * 60)
    log.info("RISK: Volatility  |  CV thresholds: %s", CV_THRESHOLDS)
    log.info("=" * 60)

    df = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = compute_volatility(df)

    out_cols = ["item_id", "store_id", "date", "sales",
                "roll_mean", "roll_std", "cv", "iqr_28",
                "volatility_regime",
                "cv_7", "cv_14", "cv_28"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_parquet(
        PROCESSED_DIR / "volatility.parquet", index=False, compression="snappy"
    )
    log.info("Volatility data saved.")

    # Also save aggregate per-series stats
    series_stats = (
        df.groupby(["item_id", "store_id"])
          .agg(mean_cv=("cv", "mean"), max_cv=("cv", "max"),
               regime=("volatility_regime", lambda x: x.mode()[0]))
          .reset_index()
    )
    series_stats.to_parquet(
        PROCESSED_DIR / "series_risk_summary.parquet", index=False
    )
    log.info("Series risk summary: %d item-store pairs", len(series_stats))

    # Write to DB
    engine = create_engine(DB_URL, pool_pre_ping=True)
    records = []
    for _, row in df.iterrows():
        date_val = row["date"]
        if hasattr(date_val, "date"):
            date_val = date_val.date()
        records.append({
            "item_id":      row["item_id"],
            "store_id":     row["store_id"],
            "date_id":      date_val,
            "flag_type":    "high_volatility" if row["volatility_regime"] == "High" else "normal_volatility",
            "anomaly_score": float(row["cv"]),
            "threshold":    CV_THRESHOLDS["medium"],
            "is_anomaly":   row["volatility_regime"] == "High",
            "rolling_cv":   float(row["cv"]),
            "rolling_mean": float(row["roll_mean"]) if pd.notna(row["roll_mean"]) else None,
            "rolling_std":  float(row["roll_std"]) if pd.notna(row["roll_std"]) else None,
        })
        if len(records) >= 100_000:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO fact_risk_flags
                        (item_id, store_id, date_id, flag_type, anomaly_score,
                         threshold, is_anomaly, rolling_cv, rolling_mean, rolling_std)
                    VALUES
                        (:item_id, :store_id, :date_id, :flag_type, :anomaly_score,
                         :threshold, :is_anomaly, :rolling_cv, :rolling_mean, :rolling_std)
                    ON CONFLICT DO NOTHING
                """), records)
            records = []
    log.info("=" * 60)


if __name__ == "__main__":
    main()
