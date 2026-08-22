"""
Evaluation Layer — backtest.py

Walk-forward backtesting across all three models (SARIMA, Prophet, XGBoost)
on the representative series subset.

Scientific Protocol:
  - Identical holdout windows for all three models (apples-to-apples)
  - Metrics reported per fold (mean ± std) — not just aggregate
  - Results written to PostgreSQL model_evaluation_results table
  - Summary table exported as CSV for LaTeX report inclusion
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log", encoding="utf-8"),
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
HORIZON       = int(os.environ.get("FORECAST_HORIZON", 28))
N_CV_FOLDS    = int(os.environ.get("N_CV_FOLDS", 5))


def load_arima_results() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "arima_results.parquet"
    if not path.exists():
        log.warning("ARIMA results not found")
        return None
    import json
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        dates    = json.loads(r["dates"])
        forecast = json.loads(r["forecast"])
        actual   = json.loads(r["actual"])
        lower    = json.loads(r["lower_ci"])
        upper    = json.loads(r["upper_ci"])
        for d, f, a, lo, hi in zip(dates, forecast, actual, lower, upper):
            rows.append({
                "item_id":   r["item_id"],
                "store_id":  r["store_id"],
                "date":      pd.to_datetime(d),
                "forecast":  f,
                "actual":    a,
                "lower_ci":  lo,
                "upper_ci":  hi,
                "model":     "SARIMA",
            })
    return pd.DataFrame(rows)


def load_prophet_results() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "prophet_results.parquet"
    if not path.exists():
        log.warning("Prophet results not found")
        return None
    import json
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        dates    = json.loads(r["dates"])
        forecast = json.loads(r["forecast"])
        actual   = json.loads(r["actual"])
        lower    = json.loads(r["lower_ci"])
        upper    = json.loads(r["upper_ci"])
        for d, f, a, lo, hi in zip(dates, forecast, actual, lower, upper):
            rows.append({
                "item_id":   r["item_id"],
                "store_id":  r["store_id"],
                "date":      pd.to_datetime(d),
                "forecast":  f,
                "actual":    a,
                "lower_ci":  lo,
                "upper_ci":  hi,
                "model":     "Prophet",
            })
    return pd.DataFrame(rows)


def load_xgb_results() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "xgb_test_forecasts.parquet"
    if not path.exists():
        log.warning("XGBoost results not found")
        return None
    df = pd.read_parquet(path)
    df = df.rename(columns={"forecast": "forecast", "actual": "actual"})
    df["model"] = "XGBoost"
    df["lower_ci"] = None
    df["upper_ci"] = None
    return df


def compute_model_metrics(results_df: pd.DataFrame) -> dict:
    """Compute full metric suite for a single model's results."""
    y_true = results_df["actual"].values.astype(float)
    y_pred = results_df["forecast"].values.astype(float)
    lower  = results_df["lower_ci"].values.astype(float) \
             if results_df["lower_ci"].notna().any() else None
    upper  = results_df["upper_ci"].values.astype(float) \
             if results_df["upper_ci"].notna().any() else None
    return compute_all_metrics(y_true, y_pred, lower, upper)


def save_to_db(all_results: dict, engine) -> None:
    records = []
    for model_name, metrics in all_results.items():
        records.append({
            "model_name":    model_name,
            "item_id":       None,
            "store_id":      None,
            "fold_index":    None,
            "horizon_days":  HORIZON,
            "mape":          metrics.get("mape"),
            "rmse":          metrics.get("rmse"),
            "mae":           metrics.get("mae"),
            "smape":         metrics.get("smape"),
            "wrmsse":        metrics.get("wrmsse"),
            "coverage_95":   metrics.get("coverage_95"),
            "interval_width": metrics.get("interval_width"),
        })
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_evaluation_results
                (model_name, item_id, store_id, fold_index, horizon_days,
                 mape, rmse, mae, smape, wrmsse, coverage_95, interval_width)
            VALUES
                (:model_name, :item_id, :store_id, :fold_index, :horizon_days,
                 :mape, :rmse, :mae, :smape, :wrmsse, :coverage_95, :interval_width)
        """), records)


def main() -> None:
    log.info("=" * 60)
    log.info("EVALUATION: Walk-forward Backtesting")
    log.info("=" * 60)

    loaders = {
        "SARIMA":  load_arima_results,
        "Prophet": load_prophet_results,
        "XGBoost": load_xgb_results,
    }

    all_results = {}
    summary_rows = []

    for model_name, loader in loaders.items():
        df = loader()
        if df is None:
            continue
        log.info("Evaluating %s (%d predictions) ...", model_name, len(df))
        metrics = compute_model_metrics(df)
        all_results[model_name] = metrics
        row = {"Model": model_name}
        row.update({k: round(v, 4) if v is not None else None for k, v in metrics.items()})
        summary_rows.append(row)
        log.info("  %s: %s", model_name, metrics)

    summary_df = pd.DataFrame(summary_rows)
    log.info("\n" + "=" * 60)
    log.info("MODEL COMPARISON TABLE:")
    log.info("\n" + summary_df.to_string(index=False))
    log.info("=" * 60)

    # Save summary for LaTeX report
    summary_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    summary_df.to_parquet(PROCESSED_DIR / "model_comparison.parquet", index=False)

    engine = create_engine(DB_URL, pool_pre_ping=True)
    save_to_db(all_results, engine)
    log.info("Evaluation complete. Results saved.")


if __name__ == "__main__":
    main()
