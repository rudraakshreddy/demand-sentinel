# Retail Demand Forecasting & Inventory Risk Monitoring Platform

> **Semester 8 Major Project** — Production-grade, 7-layer ML pipeline on the M5 Forecasting (Walmart) dataset.

---
### 🚀 **[Access the Full 2GB+ Processed Dataset & Models on Hugging Face](https://huggingface.co/datasets/snchakri/m5-retail-demand-forecasting-benchmarks)**
Because GitHub limits file sizes to 100MB, the massive engineered temporal feature matrices, Isolation Forest scores, XGBoost SHAP values, and `.pkl` artifacts are proudly hosted on Hugging Face. 
👉 `pip install datasets` and `load_dataset("snchakri/m5-retail-demand-forecasting-benchmarks")` to instantly reproduce this pipeline's outputs!
---

## Architecture Overview

```
Raw Data (M5 CSVs)
       │
       ▼
┌─────────────────┐    ┌──────────────────────┐
│  Layer 1        │    │  Layer 2             │
│  Data Ingestion │───▶│  PostgreSQL Storage  │
│  copy_raw.py    │    │  Normalised 3NF schema│
│  load_db.py     │    └──────────────────────┘
└─────────────────┘              │
                                 ▼
                    ┌──────────────────────────┐
                    │  Layer 3: ETL            │
                    │  clean.py + features.py  │
                    │  35+ engineered features  │
                    └──────────────────────────┘
                                 │
               ┌─────────────────┼──────────────────┐
               ▼                 ▼                   ▼
    ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐
    │  SARIMA      │  │  Prophet     │  │  XGBoost (Global)  │
    │  per-series  │  │  + regressors│  │  + Optuna tuning   │
    │  ADF/KPSS    │  │  + MCMC CI   │  │  + SHAP values     │
    └──────────────┘  └──────────────┘  └────────────────────┘
               │                 │                   │
               └─────────────────┼───────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  Layer 5: Risk           │
                    │  Volatility (CV, IQR)    │
                    │  Shortfall (SRE / VaR)   │
                    │  Isolation Forest        │
                    │  Stockout Risk Index     │
                    └──────────────────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                  ▼
    ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐
    │  Streamlit   │  │  Airflow DAG     │  │  LaTeX Report  │
    │  Dashboard   │  │  @daily schedule │  │  PDF submission│
    └──────────────┘  └──────────────────┘  └────────────────┘
```

---

## Quick Start

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Docker Desktop | 29+ |
| Docker Compose | v5+ |

### 2. Set up environment

```bash
# Copy and fill environment variables
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Start infrastructure (PostgreSQL + Airflow)

```bash
make airflow-up
# Airflow UI → http://localhost:8080  (airflow / airflow)
# pgAdmin    → http://localhost:5050  (admin@dfp.local / admin)
```

### 4. Ingest raw M5 data

```bash
# Set path to your M5 CSV folder (or place CSVs in Downloads/m5-forecasting-accuracy/)
make ingest
```

### 5. Run the full pipeline

```bash
make etl       # Clean + feature engineering
make train     # SARIMA, Prophet, XGBoost, Isolation Forest
make risk      # Volatility + Shortfall Risk
make evaluate  # Walk-forward backtest, metrics to DB
make serve     # Launch Streamlit dashboard on :8501
```

### 6. Run tests

```bash
make test
# → pytest tests/ -v --tb=short --cov=src
```

---

## Dataset: M5 Forecasting Competition

| File | Description | Size |
|------|-------------|------|
| `sales_train_evaluation.csv` | Daily unit sales (wide format), 30,490 items × 1,941 days | ~116 MB |
| `calendar.csv` | Date, event, SNAP flag metadata | ~101 KB |
| `sell_prices.csv` | Weekly sell prices per item-store | ~194 MB |

- **42,840 time series** (item × store)
- **5 years** of daily history (2011–2016)
- **3 states**, **10 stores**, **3 categories** (FOOD, HOBBIES, HOUSEHOLD)

---

## Model Summary

| Model | Type | Approach | Hyperparameter Selection |
|-------|------|----------|--------------------------|
| SARIMA | Per-series statistical | `pmdarima.auto_arima`, s=7 | AIC-minimising grid search |
| Prophet | Per-series probabilistic | Multiplicative seasonality + regressors | Default + changepoint_prior_scale |
| XGBoost | Global panel ML | All 42K series simultaneously | Optuna Bayesian (50 trials, TPE) |
| Isolation Forest | Anomaly detection | Rolling-stat features, contamination=2.5% | Empirical calibration |

---

## Evaluation Protocol

- **Walk-forward cross-validation** (5 folds, expanding window) — no random shuffling
- **Identical 28-day holdout** for all three forecasting models
- **Metrics reported**: MAPE, sMAPE, MAE, RMSE, WRMSSE, Coverage, Interval Width

### Scientific Rigor Checkpoints

| Check | Target | How Verified |
|-------|--------|--------------|
| Prediction interval coverage | 93–97% | `coverage_backtest()` in `shortfall.py` |
| Shortfall breach rate | ≈5% | `shortfall["shortfall_breach"].mean()` |
| Anomaly flag rate | 2–3% | `len(anomalies)/len(sales)` |
| No data leakage | All lags via `.shift()` | `test_no_future_leakage` in `test_etl.py` |

---

## Directory Structure

```
demand-forecasting-platform/
├── data/
│   ├── raw/              # M5 CSVs (copied from Downloads)
│   ├── processed/        # Parquet outputs (clean, features, forecasts, risk)
│   └── external/         # Holiday / weather augmentation (future)
├── db/
│   ├── schema.sql        # PostgreSQL DDL (8 tables, 3NF)
│   └── init_airflow_db.sh
├── src/
│   ├── ingestion/        # copy_raw.py, load_db.py
│   ├── etl/              # clean.py, features.py, pipeline.py
│   ├── models/           # arima_model.py, prophet_model.py, xgboost_model.py, isolation_forest.py
│   ├── risk/             # volatility.py, shortfall.py
│   ├── evaluation/       # metrics.py, backtest.py
│   └── serving/          # dashboard.py (Streamlit, 5 pages)
├── dags/
│   └── demand_pipeline.py  # Airflow DAG (11 tasks)
├── tests/
│   ├── test_etl.py
│   ├── test_metrics.py
│   └── test_models.py
├── report/               # LaTeX source + compiled PDF
├── docker-compose.yml    # PostgreSQL 16 + Airflow 2.9
├── requirements.txt      # Pinned versions
├── Makefile              # One-command workflow
└── README.md
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Global XGBoost panel model** | Outperforms per-series ML on M5; enables cross-series pattern learning |
| **SARIMA on representative subset (n=30)** | Full 42K × SARIMA is computationally infeasible; stratified sample preserves distribution |
| **Multiplicative Prophet seasonality** | M5 demand has proportional noise structure — multiplicative outperforms additive |
| **SRE instead of normal-distribution CI** | Historical simulation is assumption-free; mirrors financial VaR methodology |
| **Optuna TPE over grid search** | Bayesian optimisation finds better params with 10× fewer trials |
| **3×IQR clipping (not 1.5×)** | Retail demand spikes (promotions) are real signal, not measurement noise |

---

## References

1. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition. *International Journal of Forecasting*, 38(4), 1346–1364.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
4. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM 2008*.
5. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
6. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.

---

## Authors

Sem 8 Major Project · [Your Institute Name]
