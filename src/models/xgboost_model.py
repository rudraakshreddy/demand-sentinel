"""
Modeling Layer — xgboost_model.py  (FULL IMPLEMENTATION — no stubs)

Global XGBoost panel model trained on ALL M5 item-store series simultaneously.
Prediction intervals via dedicated quantile-regression models
(XGBoost 2.x  reg:quantilereg), not multiplicative approximations.

Three XGBoost models trained:
  - Point model    : reg:squarederror  (best point estimate)
  - Lower CI model : reg:quantilereg, α=0.025  (lower 95% bound)
  - Upper CI model : reg:quantilereg, α=0.975  (upper 95% bound)

GPU acceleration:
  Set XGB_DEVICE=cuda in environment to use NVIDIA GPU (e.g. RTX 5090).
  Falls back to CPU (device=cpu) when CUDA unavailable.
  tree_method='hist' is correct for both CPU and GPU in XGBoost 2.x.

References:
  Chen & Guestrin (2016). XGBoost. KDD.
  Akiba et al. (2019). Optuna. KDD.
  Koenker & Bassett (1978). Regression quantiles. Econometrica.
"""

import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON       = int(os.environ.get("FORECAST_HORIZON", 28))
N_CV_FOLDS    = int(os.environ.get("N_CV_FOLDS", 5))
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", 50))
RANDOM_SEED   = int(os.environ.get("RANDOM_SEED", 42))
LOWER_Q       = 0.025   # lower bound of 95% prediction interval
UPPER_Q       = 0.975   # upper bound of 95% prediction interval

# GPU acceleration: set XGB_DEVICE=cuda to use NVIDIA GPU (e.g. RTX 5090).
# tree_method='hist' is correct for both CPU and GPU in XGBoost 2.x.
XGB_DEVICE   = os.environ.get("XGB_DEVICE", "cpu")   # "cuda" | "cpu"
TREE_METHOD  = "hist"
log.info("XGBoost device=%s  tree_method=%s", XGB_DEVICE, TREE_METHOD)

# Feature columns — must match features.py output exactly
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "ewm_alpha_03", "ewm_alpha_01",
    "day_of_week", "day_of_month", "week_of_year", "month", "year",
    "is_weekend", "is_month_start", "is_month_end",
    "has_event", "is_sporting", "is_national", "is_religious", "is_cultural",
    "days_to_next_event",
    "sell_price", "price_change_pct", "price_vs_median", "price_vs_dept_median",
    "is_promo",
    "snap", "snap_x_weekend", "event_x_lag7", "promo_x_dow",
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss — measures quantile forecast calibration.

    L_α(y, ŷ) = (1-α)·max(ŷ-y, 0) + α·max(y-ŷ, 0)
    """
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)))


def prediction_interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float((hi - lo).mean())


# ── Walk-forward cross-validation ──────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, feature_cols: list, n_folds: int = N_CV_FOLDS) -> pd.DataFrame:
    """Walk-forward (expanding window) CV — the only valid approach for time series."""
    import xgboost as xgb
    log.info("Walk-forward CV: %d folds, horizon=%d days", n_folds, HORIZON)

    all_dates     = sorted(df["date"].unique())
    n_dates       = len(all_dates)
    min_train_days = 365
    fold_results  = []

    fold_cuts = np.linspace(min_train_days, n_dates - HORIZON - 1, n_folds, dtype=int)

    for fold_idx, split_i in enumerate(fold_cuts):
        cutoff   = all_dates[split_i]
        test_end = all_dates[min(split_i + HORIZON, n_dates - 1)]

        train_mask = df["date"] <= cutoff
        test_mask  = (df["date"] > cutoff) & (df["date"] <= test_end)

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, "sales"].values
        X_test  = df.loc[test_mask,  feature_cols].fillna(0)
        y_test  = df.loc[test_mask,  "sales"].values

        if len(y_test) == 0:
            continue

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=TREE_METHOD,
            device=XGB_DEVICE,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = np.clip(model.predict(X_test), 0, None)

        m = compute_mape(y_test, y_pred)
        r = compute_rmse(y_test, y_pred)
        log.info("  Fold %d/%d | cutoff=%s | MAPE=%.4f | RMSE=%.4f",
                 fold_idx + 1, n_folds, cutoff, m, r)

        fold_results.append({
            "fold": fold_idx + 1, "train_end": str(cutoff), "test_end": str(test_end),
            "mape": m, "rmse": r,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        })

    return pd.DataFrame(fold_results)


# ── Optuna hyperparameter optimisation ────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, X_val, y_val) -> dict:
    """Bayesian search with TPE sampler — optimises RMSE on validation fold."""
    import optuna, xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method":      TREE_METHOD,
            "device":           XGB_DEVICE,
            "random_state":     RANDOM_SEED,
            "n_jobs":           -1,
        }
        m = xgb.XGBRegressor(**params, early_stopping_rounds=20, eval_metric="rmse")
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return compute_rmse(y_val, np.clip(m.predict(X_val), 0, None))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    log.info("Best hyperparams (val RMSE=%.4f):", study.best_value)
    for k, v in study.best_params.items():
        log.info("  %-22s = %s", k, v)
    return study.best_params


# ── Train one model (point or quantile) ───────────────────────────────────────

def train_xgb_model(X_train, y_train, X_val, y_val, X_all, y_all,
                    best_params: dict, objective: str = "reg:squarederror",
                    quantile_alpha: float | None = None,
                    model_name: str = "point"):
    """Train a single XGBoost model. Uses early stopping on val, then retrains on all."""
    import xgboost as xgb

    params = dict(best_params)
    params.update({
        "objective":    objective,
        "tree_method":  TREE_METHOD,
        "device":       XGB_DEVICE,
        "random_state": RANDOM_SEED,
        "n_jobs":       -1,
    })
    if quantile_alpha is not None:
        params["quantile_alpha"] = quantile_alpha

    # Early stopping to find best n_estimators for this objective
    _eval_metric = "quantile" if quantile_alpha else "rmse"
    m_val = xgb.XGBRegressor(**params, early_stopping_rounds=30, eval_metric=_eval_metric)
    m_val.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_n = m_val.best_iteration + 1
    log.info("  [%s] best n_estimators=%d", model_name, best_n)

    # Final model: train on train+val with exact n_estimators
    params["n_estimators"] = best_n
    final = xgb.XGBRegressor(**params)
    final.fit(X_all, y_all, verbose=False)
    return final


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("MODEL: XGBoost (Global Panel)  horizon=%d  device=%s",
             HORIZON, XGB_DEVICE)
    log.info("=" * 60)

    feat_path = PROCESSED_DIR / "sales_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            "Run etl/features.py first: data/processed/sales_features.parquet not found"
        )

    log.info("Loading feature matrix ...")
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(avail)
    log.info("Using %d / %d feature columns", len(avail), len(FEATURE_COLS))
    if missing:
        log.warning("Missing features (filled with 0): %s", sorted(missing))

    # ── Walk-forward CV ────────────────────────────────────────────────────────
    cv_results = walk_forward_cv(df, avail, n_folds=N_CV_FOLDS)
    log.info("CV Mean MAPE=%.4f±%.4f  RMSE=%.4f±%.4f",
             cv_results["mape"].mean(), cv_results["mape"].std(),
             cv_results["rmse"].mean(), cv_results["rmse"].std())
    cv_results.to_parquet(PROCESSED_DIR / "xgb_cv_results.parquet", index=False)

    # ── Time-aware splits: train / val / test ─────────────────────────────────
    all_dates  = sorted(df["date"].unique())
    test_start = all_dates[-HORIZON]
    val_start  = all_dates[-(HORIZON * 2)]

    tr_m  = df["date"] < val_start
    va_m  = (df["date"] >= val_start) & (df["date"] < test_start)
    te_m  = df["date"] >= test_start
    all_m = df["date"] < test_start   # train+val for final fit

    X_tr, y_tr = df.loc[tr_m,  avail].fillna(0), df.loc[tr_m,  "sales"].values
    X_va, y_va = df.loc[va_m,  avail].fillna(0), df.loc[va_m,  "sales"].values
    X_te, y_te = df.loc[te_m,  avail].fillna(0), df.loc[te_m,  "sales"].values
    X_al, y_al = df.loc[all_m, avail].fillna(0), df.loc[all_m, "sales"].values
    log.info("Split — train: %d  val: %d  test: %d", len(y_tr), len(y_va), len(y_te))

    # ── Hyperparameter tuning (on point objective) ─────────────────────────────
    log.info("Bypassing Optuna tuning: using best params from full run.")
    t0 = time.perf_counter()
    best_params = {
        "n_estimators": 602,
        "learning_rate": 0.017247061024071843,
        "max_depth": 10,
        "subsample": 0.5359498404978726,
        "colsample_bytree": 0.6668793407545431,
        "min_child_weight": 17,
        "reg_alpha": 0.2054844384852384,
        "reg_lambda": 4.861052139411129e-06,
        "device": XGB_DEVICE,
        "tree_method": "hist"
    }
    log.info("  Tuning bypassed: %.0fs", time.perf_counter() - t0)

    # ── Train 3 models: point + lower CI + upper CI ────────────────────────────
    log.info("Training point model ...")
    t0 = time.perf_counter()
    m_point = train_xgb_model(X_tr, y_tr, X_va, y_va, X_al, y_al, best_params,
                               objective="reg:squarederror", model_name="point")
    log.info("  Point: %.0fs", time.perf_counter() - t0)

    log.info("Training lower CI (q=%.3f) model ...", LOWER_Q)
    t0 = time.perf_counter()
    m_lower = train_xgb_model(X_tr, y_tr, X_va, y_va, X_al, y_al, best_params,
                               objective="reg:quantileerror",
                               quantile_alpha=LOWER_Q, model_name=f"lower_q{LOWER_Q}")
    log.info("  Lower CI: %.0fs", time.perf_counter() - t0)

    log.info("Training upper CI (q=%.3f) model ...", UPPER_Q)
    t0 = time.perf_counter()
    m_upper = train_xgb_model(X_tr, y_tr, X_va, y_va, X_al, y_al, best_params,
                               objective="reg:quantileerror",
                               quantile_alpha=UPPER_Q, model_name=f"upper_q{UPPER_Q}")
    log.info("  Upper CI: %.0fs", time.perf_counter() - t0)

    # ── Predict on test set ────────────────────────────────────────────────────
    y_pred  = np.clip(m_point.predict(X_te), 0, None)
    y_lower = np.clip(m_lower.predict(X_te), 0, None)
    y_upper = np.clip(m_upper.predict(X_te), 0, None)

    # Guarantee monotonicity: lower ≤ point ≤ upper
    y_lower = np.minimum(y_lower, y_pred)
    y_upper = np.maximum(y_upper, y_pred)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    mape_v   = compute_mape(y_te, y_pred)
    rmse_v   = compute_rmse(y_te, y_pred)
    coverage = prediction_interval_coverage(y_te, y_lower, y_upper)
    width    = mean_interval_width(y_lower, y_upper)
    pb_lo    = pinball_loss(y_te, y_lower, LOWER_Q)
    pb_hi    = pinball_loss(y_te, y_upper, UPPER_Q)

    log.info("=" * 60)
    log.info("Test Metrics:")
    log.info("  MAPE          : %.4f", mape_v)
    log.info("  RMSE          : %.4f", rmse_v)
    log.info("  Coverage 95%%  : %.4f  (target 0.9500)", coverage)
    log.info("  Mean Width    : %.4f", width)
    log.info("  Pinball lower : %.4f", pb_lo)
    log.info("  Pinball upper : %.4f", pb_hi)
    log.info("=" * 60)

    # ── SHAP explainability (TreeExplainer — exact, not approximate) ───────────
    try:
        import shap
        log.info("Computing SHAP values (n=2000 test sample) ...")
        t0 = time.perf_counter()
        rng       = np.random.default_rng(RANDOM_SEED)
        idx       = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
        X_samp    = X_te.iloc[idx]
        explainer = shap.TreeExplainer(m_point)
        sv        = explainer.shap_values(X_samp)
        shap_df   = pd.DataFrame(sv, columns=avail)
        shap_df.to_parquet(PROCESSED_DIR / "xgb_shap_values.parquet", index=False)
        log.info("  SHAP: %.0fs", time.perf_counter() - t0)
        mean_abs = np.abs(sv).mean(axis=0)
        imp = sorted(zip(avail, mean_abs), key=lambda x: -x[1])
        log.info("Top 10 SHAP features:")
        for feat, val in imp[:10]:
            log.info("  %-35s  %.6f", feat, val)
        pd.DataFrame(imp, columns=["feature", "mean_abs_shap"]).to_csv(
            PROCESSED_DIR / "xgb_shap_importance.csv", index=False
        )
    except Exception as e:
        log.warning("SHAP computation failed or shap not installed: %s", e)

    # ── Save models ────────────────────────────────────────────────────────────
    for name, model in [("point", m_point), ("lower", m_lower), ("upper", m_upper)]:
        with open(MODEL_DIR / f"xgboost_{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(MODEL_DIR / "xgboost_features.json", "w") as f:
        json.dump(avail, f)
    log.info("Models saved → %s", MODEL_DIR)

    # ── Save test forecasts ────────────────────────────────────────────────────
    test_df = df.loc[te_m, ["item_id", "store_id", "date"]].copy()
    test_df["forecast"]    = y_pred
    test_df["actual"]      = y_te
    test_df["lower_ci_95"] = y_lower
    test_df["upper_ci_95"] = y_upper
    test_df["model_name"]  = "XGBoost"
    test_df.to_parquet(PROCESSED_DIR / "xgb_test_forecasts.parquet", index=False)

    # ── Save summary metrics ───────────────────────────────────────────────────
    summary = {
        "mape": round(mape_v, 4), "rmse": round(rmse_v, 4),
        "coverage_95": round(coverage, 4), "interval_width": round(width, 4),
        "pinball_lower": round(pb_lo, 4), "pinball_upper": round(pb_hi, 4),
        "cv_mape_mean": round(cv_results["mape"].mean(), 4),
        "cv_mape_std":  round(cv_results["mape"].std(),  4),
        "cv_rmse_mean": round(cv_results["rmse"].mean(), 4),
        "cv_rmse_std":  round(cv_results["rmse"].std(),  4),
    }
    with open(PROCESSED_DIR / "xgb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary: %s", json.dumps(summary, indent=2))

    # ── Persist to PostgreSQL ──────────────────────────────────────────────────
    try:
        from sqlalchemy import create_engine, text
        engine  = create_engine(DB_URL, pool_pre_ping=True)
        records = [
            {"item_id": r["item_id"], "store_id": r["store_id"],
             "date_id": r["date"].date(), "model_name": "XGBoost",
             "forecast": float(r["forecast"]), "lower_ci_95": float(r["lower_ci_95"]),
             "upper_ci_95": float(r["upper_ci_95"])}
            for _, r in test_df.iterrows()
        ]
        chunk = 10_000
        for i in range(0, len(records), chunk):
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO fact_forecasts
                        (item_id, store_id, date_id, model_name,
                         forecast, lower_ci_95, upper_ci_95)
                    VALUES
                        (:item_id, :store_id, :date_id, :model_name,
                         :forecast, :lower_ci_95, :upper_ci_95)
                    ON CONFLICT (item_id, store_id, date_id, model_name) DO UPDATE
                        SET forecast     = EXCLUDED.forecast,
                            lower_ci_95  = EXCLUDED.lower_ci_95,
                            upper_ci_95  = EXCLUDED.upper_ci_95
                """), records[i: i + chunk])
        log.info("Saved %d XGBoost rows to PostgreSQL.", len(records))
    except Exception as exc:
        log.warning("PostgreSQL write skipped (DB may not be running): %s", exc)

    log.info("=" * 60)
    log.info("XGBoost training COMPLETE.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
