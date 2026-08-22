"""
ETL Layer — clean.py

Applies systematic data quality rules to the M5 raw sales data.

Scientific Rigor:
  - Imputation strategy documented per-column with justification
  - Outlier clipping at item-store level (not global) to preserve
    heterogeneous demand distributions across SKUs
  - All transformations are logged with row-level statistics
  - Cleaned data saved to parquet for fast downstream access
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/etl.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_sales_long() -> pd.DataFrame:
    """Load and melt the wide-format sales CSV to long format."""
    log.info("Reading sales_train_evaluation.csv ...")
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    id_cols   = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    date_cols = [c for c in sales.columns if c.startswith("d_")]

    # Build day → date map
    cal = pd.read_csv(RAW_DIR / "calendar.csv", usecols=["d", "date"])
    day_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))

    log.info("Melting %d items × %d days ...", len(sales), len(date_cols))
    long = sales[id_cols + date_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    long["date"] = long["d"].map(day_to_date)
    long = long.drop(columns=["d", "id"])
    long = long.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    log.info("  Long-format shape: %s", long.shape)
    return long


def impute_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy:
      - NaN sales → 0 (store closed / item unlisted; not missing at random)
      - Negative sales → 0 (data entry error / returns)
    Justification: M5 competition rules define sales as non-negative integers.
    """
    n_null = df["sales"].isna().sum()
    n_neg  = (df["sales"] < 0).sum()
    log.info("  Nulls in sales: %d → filled with 0", n_null)
    log.info("  Negatives in sales: %d → clipped to 0", n_neg)
    df["sales"] = df["sales"].fillna(0).clip(lower=0)
    return df


def clip_outliers_iqr(df: pd.DataFrame, col: str = "sales",
                      multiplier: float = 3.0) -> pd.DataFrame:
    """
    IQR-based outlier clipping at the (item_id, store_id) level.

    Uses a 3×IQR fence (more conservative than 1.5× to preserve genuine
    demand spikes which are economically real and not measurement errors).

    Upper fence = Q3 + multiplier * IQR per series.
    Lower fence = 0 (sales can't be negative).
    """
    log.info("Clipping outliers at item-store level (multiplier=%.1f × IQR) ...", multiplier)
    original_max = df[col].max()

    def clip_group(grp: pd.DataFrame) -> pd.DataFrame:
        q1  = grp[col].quantile(0.25)
        q3  = grp[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + multiplier * iqr
        grp[col] = grp[col].clip(upper=upper)
        return grp

    df = df.groupby(["item_id", "store_id"], group_keys=False).apply(clip_group)
    log.info("  Max sales before clip: %.0f  after: %.0f",
             original_max, df[col].max())
    return df


def load_and_merge_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join sell prices using wm_yr_wk from calendar."""
    log.info("Merging sell prices ...")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    cal    = pd.read_csv(RAW_DIR / "calendar.csv",
                         usecols=["date", "wm_yr_wk"],
                         parse_dates=["date"])
    df = df.merge(cal[["date", "wm_yr_wk"]], on="date", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Forward-fill price within each item-store series
    df = df.sort_values(["item_id", "store_id", "date"])
    df["sell_price"] = df.groupby(["item_id", "store_id"])["sell_price"].ffill().bfill()
    n_null_price = df["sell_price"].isna().sum()
    log.info("  Remaining null prices after ffill/bfill: %d", n_null_price)
    return df


def merge_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar attributes: events, SNAP flags."""
    log.info("Merging calendar features ...")
    cal = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    cal.columns = cal.columns.str.lower()   # normalise: snap_CA -> snap_ca
    keep = ["date", "event_name_1", "event_type_1",
            "event_name_2", "event_type_2",
            "snap_ca", "snap_tx", "snap_wi"]
    cal = cal[[c for c in keep if c in cal.columns]]
    df = df.merge(cal, on="date", how="left")
    return df



def save_parquet(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    log.info("  Saved: %s  (%.1f MB)",
             path, path.stat().st_size / 1e6)


def main() -> None:
    log.info("=" * 60)
    log.info("ETL CLEAN  |  M5 Demand Forecasting Platform")
    log.info("=" * 60)
    df = load_sales_long()
    df = impute_sales(df)
    df = clip_outliers_iqr(df)
    df = load_and_merge_prices(df)
    df = merge_calendar_features(df)

    # Add SNAP flag per state
    def snap_flag(row):
        state = row["state_id"]
        if state == "CA": return bool(row.get("snap_ca", 0))
        if state == "TX": return bool(row.get("snap_tx", 0))
        if state == "WI": return bool(row.get("snap_wi", 0))
        return False
    df["snap"] = df.apply(snap_flag, axis=1)

    log.info("Final cleaned shape: %s", df.shape)
    log.info("Date range: %s → %s", df["date"].min(), df["date"].max())
    save_parquet(df, "sales_clean")
    log.info("=" * 60)
    log.info("Clean ETL complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
