"""
Airflow DAG — demand_pipeline.py

Full 7-stage orchestration pipeline for the Retail Demand Forecasting Platform.

Stages:
  1. ingest_data        — Copy raw M5 files, load to PostgreSQL
  2. validate_schema    — Assert row counts, null fractions, date ranges
  3. run_etl            — Clean + feature engineering
  4. train_sarima       — Per-series SARIMA models
  5. train_prophet      — Per-series Prophet models
  6. train_xgboost      — Global XGBoost panel model
  7. train_iso_forest   — Isolation Forest anomaly detection
  8. compute_volatility — Rolling CV and regime classification
  9. compute_shortfall  — SRE / VaR and SRI composite score
 10. evaluate_models    — Walk-forward backtest, write metrics to DB
 11. notify_complete    — Log summary, flag counts, coverage stats

Schedule: @daily (retrain on new data when available)
"""

from datetime import datetime, timedelta
import logging
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "demand-forecasting-platform",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="demand_forecasting_pipeline",
    default_args=DEFAULT_ARGS,
    description="Retail Demand Forecasting — Full 7-Layer Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["forecasting", "risk", "ml", "m5"],
) as dag:

    # ── Task functions ─────────────────────────────────────────────────────────

    def task_ingest(**context):
        """Copy raw data and load into PostgreSQL."""
        from src.ingestion.copy_raw import main as copy_raw
        from src.ingestion.load_db import main as load_db
        log.info("=== STAGE 1: Ingestion ===")
        copy_raw()
        load_db()
        _log_pipeline_stage(context, "ingest_data", "success")

    def task_validate(**context):
        """Schema and quality validation gates."""
        import pandas as pd
        from pathlib import Path
        from sqlalchemy import create_engine, text

        log.info("=== STAGE 2: Schema Validation ===")
        engine = create_engine(_db_url())

        with engine.connect() as conn:
            # Gate 1: calendar rows
            cal_count = conn.execute(text("SELECT COUNT(*) FROM dim_calendar")).scalar()
            assert cal_count >= 1941, f"Calendar rows too low: {cal_count}"
            log.info("  dim_calendar: %d rows ✓", cal_count)

            # Gate 2: item count
            item_count = conn.execute(text("SELECT COUNT(*) FROM dim_item")).scalar()
            assert item_count >= 3000, f"Item count too low: {item_count}"
            log.info("  dim_item: %d rows ✓", item_count)

            # Gate 3: sales rows
            sales_count = conn.execute(text("SELECT COUNT(*) FROM fact_sales")).scalar()
            assert sales_count >= 10_000_000, f"Sales rows too low: {sales_count}"
            log.info("  fact_sales: %d rows ✓", sales_count)

            # Gate 4: null fraction in sales
            null_pct = conn.execute(
                text("SELECT AVG(CASE WHEN sales IS NULL THEN 1.0 ELSE 0.0 END) FROM fact_sales")
            ).scalar()
            assert null_pct < 0.01, f"Null fraction in sales too high: {null_pct:.4f}"
            log.info("  Null fraction in sales: %.4f ✓", null_pct)

        _log_pipeline_stage(context, "validate_schema", "success", rows_processed=sales_count)

    def task_etl(**context):
        """Run full ETL: clean → feature engineering."""
        from src.etl.pipeline import run_pipeline
        log.info("=== STAGE 3: ETL ===")
        run_pipeline()
        _log_pipeline_stage(context, "run_etl", "success")

    def task_train_sarima(**context):
        from src.models.arima_model import main
        log.info("=== STAGE 4: SARIMA Training ===")
        main()
        _log_pipeline_stage(context, "train_sarima", "success")

    def task_train_prophet(**context):
        from src.models.prophet_model import main
        log.info("=== STAGE 5: Prophet Training ===")
        main()
        _log_pipeline_stage(context, "train_prophet", "success")

    def task_train_xgboost(**context):
        from src.models.xgboost_model import main
        log.info("=== STAGE 6: XGBoost Training ===")
        main()
        _log_pipeline_stage(context, "train_xgboost", "success")

    def task_train_isoforest(**context):
        from src.models.isolation_forest import main
        log.info("=== STAGE 7: Isolation Forest ===")
        main()
        _log_pipeline_stage(context, "train_iso_forest", "success")

    def task_volatility(**context):
        from src.risk.volatility import main
        log.info("=== STAGE 8: Volatility Computation ===")
        main()
        _log_pipeline_stage(context, "compute_volatility", "success")

    def task_shortfall(**context):
        from src.risk.shortfall import main
        log.info("=== STAGE 9: Shortfall Risk (SRE) ===")
        main()
        _log_pipeline_stage(context, "compute_shortfall", "success")

    def task_evaluate(**context):
        from src.evaluation.backtest import main
        log.info("=== STAGE 10: Model Evaluation ===")
        main()
        _log_pipeline_stage(context, "evaluate_models", "success")

    def task_notify(**context):
        """Final summary notification: log key stats."""
        from pathlib import Path
        import pandas as pd

        log.info("=== STAGE 11: Pipeline Complete ===")
        comparison_path = Path("data/processed/model_comparison.csv")
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            log.info("\n" + "=" * 60)
            log.info("MODEL COMPARISON SUMMARY:")
            log.info(df.to_string(index=False))
            log.info("=" * 60)

        anomalies_path = Path("data/processed/anomalies.parquet")
        if anomalies_path.exists():
            anoms = pd.read_parquet(anomalies_path)
            log.info("Total anomaly flags: %d", len(anoms))

        _log_pipeline_stage(context, "notify_complete", "success")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _db_url() -> str:
        return (
            f"postgresql+psycopg2://"
            f"{os.environ.get('DB_USER', 'forecast_user')}:"
            f"{os.environ.get('DB_PASS', 'forecast_pass')}@"
            f"{os.environ.get('DB_HOST', 'postgres')}:"
            f"{os.environ.get('DB_PORT', '5432')}/"
            f"{os.environ.get('DB_NAME', 'demand_forecasting')}"
        )

    def _log_pipeline_stage(context: dict, stage: str, status: str,
                             rows_processed: int = None) -> None:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(_db_url())
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO pipeline_run_log
                        (dag_run_id, stage, status, rows_processed)
                    VALUES
                        (:dag_run_id, :stage, :status, :rows_processed)
                """), {
                    "dag_run_id":       context.get("run_id", "manual"),
                    "stage":            stage,
                    "status":           status,
                    "rows_processed":   rows_processed,
                })
        except Exception as e:
            log.warning("Failed to log pipeline stage: %s", e)

    # ── Operators ──────────────────────────────────────────────────────────────

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest,
    )

    validate = PythonOperator(
        task_id="validate_schema",
        python_callable=task_validate,
    )

    etl = PythonOperator(
        task_id="run_etl",
        python_callable=task_etl,
    )

    train_sarima = PythonOperator(
        task_id="train_sarima",
        python_callable=task_train_sarima,
    )

    train_prophet = PythonOperator(
        task_id="train_prophet",
        python_callable=task_train_prophet,
    )

    train_xgboost = PythonOperator(
        task_id="train_xgboost",
        python_callable=task_train_xgboost,
    )

    train_isoforest = PythonOperator(
        task_id="train_isolation_forest",
        python_callable=task_train_isoforest,
    )

    volatility = PythonOperator(
        task_id="compute_volatility",
        python_callable=task_volatility,
    )

    shortfall = PythonOperator(
        task_id="compute_shortfall",
        python_callable=task_shortfall,
    )

    evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=task_evaluate,
        trigger_rule=TriggerRule.ALL_DONE,  # run even if some models failed
    )

    notify = PythonOperator(
        task_id="notify_complete",
        python_callable=task_notify,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    end = EmptyOperator(task_id="end")

    # ── DAG wiring ─────────────────────────────────────────────────────────────
    # Linear: ingest → validate → etl → parallel training → risk → evaluate → notify

    start >> ingest >> validate >> etl

    # Parallel model training (independent of each other)
    etl >> [train_sarima, train_prophet, train_xgboost, train_isoforest]

    # Risk layer depends on XGBoost (for SRE) and IsoForest
    [train_xgboost, train_isoforest] >> volatility
    [train_xgboost, train_isoforest] >> shortfall

    # Evaluation waits for all models + risk
    [train_sarima, train_prophet, train_xgboost,
     volatility, shortfall] >> evaluate

    evaluate >> notify >> end
