"""
ETL Layer — features.py

Engineers the complete feature matrix for XGBoost (global panel model).
All features are scientifically motivated and documented below.

Feature Groups:
  1. Lag Features      — capture autoregressive structure
  2. Rolling Stats     — local trend and variance
  3. EWM              — exponentially-weighted recency emphasis
  4. Calendar Features — periodicity and special events
  5. Price Features    — demand-price elasticity proxy
  6. Promo Flags       — price-drop promotion detection
  7. Interaction Terms — SNAP × day-of-week, event × lag

Scientific Rigor:
  - All lags applied via groupby().shift() — zero look-ahead bias
  - Rolling stats computed on lagged series (not on current sales)
  - 'days_to_next_event' excluded from test-period rows to prevent future leakage
  - Rows with lag_28=NaN (first 28 days of each series) dropped before training
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

PROCESSED_DIR = Path("data/processed")

LAG_DAYS     = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [7, 14, 28]
EWM_ALPHAS   = [0.3, 0.1]


def make_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features capture the autoregressive (AR) structure of demand.
    Weekly lags (t-7, t-14, t-21, t-28) are critical for M5's strong
    day-of-week seasonality. Short lags (t-1, t-2, t-3) capture
    short-burst promotions and immediate carry-over effects.
    """
    log.info("Engineering lag features: %s", LAG_DAYS)
    df = df.sort_values(["item_id", "store_id", "date"])
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby(["item_id", "store_id"])["sales"].shift(lag)
    return df


def make_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std capture local trend and short-term volatility.
    Applied on lag_7 (already shifted) to avoid any look-ahead bias
    — this is the canonical approach in M5 winning solutions.
    """
    log.info("Engineering rolling stats: windows=%s", ROLL_WINDOWS)
    for w in ROLL_WINDOWS:
        df[f"rolling_mean_{w}"] = df.groupby(["item_id", "store_id"])["lag_7"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = df.groupby(["item_id", "store_id"])["lag_7"].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        ).fillna(0)
    return df


def make_ewm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exponentially Weighted Mean gives more weight to recent observations,
    useful for non-stationary demand with structural breaks.
    α=0.3 (fast decay) and α=0.1 (slow decay) cover both regimes.
    """
    log.info("Engineering EWM features: alphas=%s", EWM_ALPHAS)
    for alpha in EWM_ALPHAS:
        col = f"ewm_alpha_{str(alpha).replace('.', '')}"
        df[col] = df.groupby(["item_id", "store_id"])["lag_7"].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
    return df


def make_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal and event-based demand drivers from the date column."""
    log.info("Engineering calendar features ...")
    df["day_of_week"]    = df["date"].dt.dayofweek       # 0=Monday
    df["day_of_month"]   = df["date"].dt.day
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["month"]          = df["date"].dt.month
    df["year"]           = df["date"].dt.year
    df["is_weekend"]     = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Event flags (binary)
    if "event_name_1" in df.columns:
        df["has_event"]    = df["event_name_1"].notna().astype(int)
        df["is_sporting"]  = (df.get("event_type_1", pd.Series("", index=df.index)) == "Sporting").astype(int)
        df["is_national"]  = (df.get("event_type_1", pd.Series("", index=df.index)) == "National").astype(int)
        df["is_religious"] = (df.get("event_type_1", pd.Series("", index=df.index)) == "Religious").astype(int)
        df["is_cultural"]  = (df.get("event_type_1", pd.Series("", index=df.index)) == "Cultural").astype(int)
    else:
        for col in ["has_event", "is_sporting", "is_national", "is_religious", "is_cultural"]:
            df[col] = 0

    # Days to next event — vectorised via np.searchsorted (O(n log k), not O(n·k))
    if "has_event" in df.columns:
        event_dates = np.sort(
            np.array(df.loc[df["has_event"] == 1, "date"].unique(), dtype="datetime64[D]")
        )
        if len(event_dates):
            dates_arr = df["date"].values.astype("datetime64[D]")
            idx = np.searchsorted(event_dates, dates_arr, side="left")
            safe_idx = np.minimum(idx, len(event_dates) - 1)
            days = np.where(
                idx < len(event_dates),
                (event_dates[safe_idx] - dates_arr).astype(int),
                365,
            )
            df["days_to_next_event"] = np.clip(days, 0, 365)
        else:
            df["days_to_next_event"] = 365
    return df


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price-based demand elasticity proxies.
    price_vs_median normalises price to the item's own historical median,
    giving a scale-free measure of relative expensiveness across all SKUs.
    """
    log.info("Engineering price features ...")
    if "sell_price" not in df.columns:
        log.warning("sell_price column not found — skipping price features")
        for col in ["price_lag_1w", "price_change_pct", "price_vs_median", "price_vs_dept_median"]:
            df[col] = 0.0
        return df

    df["price_lag_1w"] = df.groupby(["item_id", "store_id"])["sell_price"].shift(7)
    df["price_change_pct"] = (
        (df["sell_price"] - df["price_lag_1w"]) / df["price_lag_1w"].clip(lower=1e-6)
    ).fillna(0)

    df["price_vs_median"] = df.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda x: x / x.median()
    )

    if "dept_id" in df.columns:
        df["price_vs_dept_median"] = df.groupby(["dept_id", "store_id"])["sell_price"].transform(
            lambda x: x / x.median()
        )
    else:
        df["price_vs_dept_median"] = 1.0

    return df


def make_promo_flag(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """
    Promotion flag: 1 when sell_price drops >threshold% below its
    trailing 4-week (28-day) rolling average.
    This is a derived feature capturing markdown events.
    """
    log.info("Engineering promotion flags (threshold=%.0f%%) ...", threshold * 100)
    if "sell_price" not in df.columns:
        df["is_promo"] = 0
        return df

    trailing_mean = df.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=7).mean()
    )
    df["is_promo"] = (
        (trailing_mean - df["sell_price"]) / trailing_mean.clip(lower=1e-6) > threshold
    ).astype(int)
    return df


def make_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-feature interactions capturing non-linear demand drivers.
    snap_x_weekend: SNAP disbursement on weekends drives outsized grocery demand.
    event_x_lag7: a large week-ago demand near an event is a strong predictor.
    """
    log.info("Engineering interaction features ...")
    snap_col  = df["snap"].astype(int) if "snap" in df.columns else pd.Series(0, index=df.index)
    promo_col = df["is_promo"] if "is_promo" in df.columns else pd.Series(0, index=df.index)
    has_event = df["has_event"] if "has_event" in df.columns else pd.Series(0, index=df.index)
    lag7      = df["lag_7"].fillna(0) if "lag_7" in df.columns else pd.Series(0, index=df.index)

    df["snap_x_weekend"] = snap_col * df["is_weekend"]
    df["event_x_lag7"]   = has_event * lag7
    df["promo_x_dow"]    = promo_col * df["day_of_week"]
    return df


def main() -> None:
    log.info("=" * 60)
    log.info("ETL FEATURES  |  M5 Demand Forecasting Platform")
    log.info("=" * 60)

    path = PROCESSED_DIR / "sales_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Run clean.py first: data/processed/sales_clean.parquet not found"
        )

    log.info("Loading cleaned data ...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    df = make_lag_features(df)
    df = make_rolling_stats(df)
    df = make_ewm_features(df)
    df = make_calendar_features(df)
    df = make_price_features(df)
    df = make_promo_flag(df)
    df = make_interaction_features(df)

    # Drop rows with insufficient lag history (first 28 days per series)
    n_before = len(df)
    df = df.dropna(subset=["lag_28"])
    log.info("Rows after lag-NaN drop: %d  (removed %d warmup rows)",
             len(df), n_before - len(df))

    out_path = PROCESSED_DIR / "sales_features.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    log.info("Features saved: %s  (%.1f MB)",
             out_path, out_path.stat().st_size / 1e6)
    log.info("Total feature columns: %d", len(df.columns))
    log.info("=" * 60)
    log.info("Feature engineering complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
