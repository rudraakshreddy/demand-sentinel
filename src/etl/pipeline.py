"""
ETL Layer — pipeline.py

Orchestrates the full ETL sequence: clean → features.
Can be run standalone or called from the Airflow DAG.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("ETL PIPELINE START")
    log.info("=" * 60)

    from src.etl.clean import main as run_clean
    from src.etl.features import main as run_features

    log.info("Step 1/2: Data cleaning ...")
    run_clean()

    log.info("Step 2/2: Feature engineering ...")
    run_features()

    log.info("=" * 60)
    log.info("ETL PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
