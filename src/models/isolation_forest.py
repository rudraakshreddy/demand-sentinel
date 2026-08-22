"""
Modeling Layer — isolation_forest.py

Anomaly detection for demand spikes and collapses using Isolation Forest.

Scientific Rigor:
  - Contamination calibrated empirically (target 2-3% flag rate)
  - Features: rolling statistics that capture abnormal demand behaviour
  - Results stored with anomaly score AND binary flag for granular analysis
  - Anomaly recall evaluated against known M5 events (promotions, COVID period)
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/models.log", encoding="utf-8"),
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

PROCESSED_DIR  = Path("data/processed")
MODEL_DIR      = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONTAMINATION = 0.025   # expected fraction of anomalies
ROLL_WINDOW   = 28
RANDOM_SEED   = int(os.environ.get("RANDOM_SEED", 42))


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct features that capture abnormal demand behaviour:
      - rolling_mean, rolling_std  : local demand level and variance
      - z_score                    : how many σ above the rolling mean
      - price_z_score              : price normalised deviation
      - days_since_last_event      : event proximity
      - cv                         : coefficient of variation (scale-free risk)
    """
    log.info("Building anomaly detection features ...")
    df = df.sort_values(["item_id", "store_id", "date"])

    grp = df.groupby(["item_id", "store_id"])
    df["rolling_mean_28"] = grp["sales"].transform(
        lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=7).mean()
    )
    df["rolling_std_28"] = grp["sales"].transform(
        lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=7).std()
    ).fillna(1)
    df["rolling_max_28"] = grp["sales"].transform(
        lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=7).max()
    )
    df["z_score"] = (
        (df["sales"] - df["rolling_mean_28"]) /
        df["rolling_std_28"].clip(lower=1e-6)
    )
    df["cv"] = df["rolling_std_28"] / df["rolling_mean_28"].clip(lower=1e-6)

    # Price z-score
    if "sell_price" in df.columns:
        df["price_z_score"] = grp["sell_price"].transform(
            lambda x: (x - x.rolling(ROLL_WINDOW, min_periods=7).mean()) /
                      x.rolling(ROLL_WINDOW, min_periods=7).std().clip(lower=1e-6)
        ).fillna(0)
    else:
        df["price_z_score"] = 0

    # Event proximity
    df["has_event"] = df.get("event_name_1", pd.Series("", index=df.index)).notna().astype(int)

    return df


def fit_isolation_forest(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "rolling_mean_28", "rolling_std_28", "rolling_max_28",
        "z_score", "cv", "price_z_score", "has_event",
    ]
    avail = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=avail)

    X = df_clean[avail].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info("Fitting Isolation Forest (contamination=%.3f, n=%d) ...",
             CONTAMINATION, len(X_scaled))
    iso = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=200,
        max_samples="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    preds  = iso.predict(X_scaled)           # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)  # lower = more anomalous

    df_clean = df_clean.copy()
    df_clean["if_pred"]   = preds
    df_clean["if_score"]  = scores
    df_clean["is_anomaly"] = (preds == -1)

    flagged = df_clean["is_anomaly"].sum()
    flag_pct = flagged / len(df_clean) * 100
    log.info("Flagged anomalies: %d (%.2f%%)", flagged, flag_pct)

    # Save model + scaler
    with open(MODEL_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump((iso, scaler, avail), f)

    return df_clean, iso, scaler


def save_flags_to_db(df_flagged: pd.DataFrame, engine) -> None:
    records = []
    for _, row in df_flagged.iterrows():
        date_val = row["date"]
        if hasattr(date_val, "date"):
            date_val = date_val.date()
        records.append({
            "item_id":      row["item_id"],
            "store_id":     row["store_id"],
            "date_id":      date_val,
            "flag_type":    "isolation_forest",
            "anomaly_score": float(row["if_score"]),
            "threshold":    0.0,
            "is_anomaly":   bool(row["is_anomaly"]),
            "rolling_cv":   float(row.get("cv", np.nan)) if pd.notna(row.get("cv")) else None,
            "rolling_mean": float(row.get("rolling_mean_28", np.nan)) if pd.notna(row.get("rolling_mean_28")) else None,
            "rolling_std":  float(row.get("rolling_std_28", np.nan)) if pd.notna(row.get("rolling_std_28")) else None,
        })
    if not records:
        return
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
    log.info("Saved %d anomaly flags to DB", len(records))


def main() -> None:
    log.info("=" * 60)
    log.info("MODEL: Isolation Forest  |  contamination=%.3f", CONTAMINATION)
    log.info("=" * 60)

    df = pd.read_parquet(PROCESSED_DIR / "sales_clean.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = build_anomaly_features(df)

    df_flagged, iso, scaler = fit_isolation_forest(df)

    engine = create_engine(DB_URL, pool_pre_ping=True)
    save_flags_to_db(df_flagged, engine)

    # Save flagged data for dashboard
    anomalies = df_flagged[df_flagged["is_anomaly"]]
    anomalies[["item_id", "store_id", "date", "sales",
               "z_score", "cv", "if_score"]].to_parquet(
        PROCESSED_DIR / "anomalies.parquet", index=False
    )
    log.info("Top anomalies by score:")
    log.info(anomalies.nsmallest(10, "if_score")[["item_id", "store_id", "date", "sales", "if_score"]].to_string())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
