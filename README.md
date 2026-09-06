# Demand Sentinel

**Global gradient-boosted forecasting for intermittent retail demand.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-live-FF4B4B.svg)](https://demand-sentinel.streamlit.app)

**[Live dashboard](https://demand-sentinel.streamlit.app)** · **[Full technical paper (PDF)](report/demand_sentinel_report.pdf)** · **[Processed dataset on Hugging Face](https://huggingface.co/datasets/snchakri/m5-retail-demand-forecasting-benchmarks)**

---

## What this is

An end-to-end demand forecasting and inventory-risk platform built on the **M5 / Walmart** dataset:
**30,490 store–item series**, 1,941 days, three US states, ten stores, 3,049 products.

The defining property of this data is **intermittency**. At store–item granularity, **68.0 %** of all
series–day observations are exactly zero and mean daily sales are **1.13 units**. That single fact
determines both which models are admissible and — less obviously — which accuracy metrics are meaningful.

## The two decisions that define the approach

**1. A Tweedie objective, not squared error.**

The response is a non-negative count that is zero most of the time. The Tweedie family with variance power
`1 < p < 2` is a compound Poisson–gamma distribution: a point mass at exactly zero plus a continuous
positive density. That is the shape of the data. Squared-error regression instead assumes a symmetric
Gaussian error around a possibly negative mean — distributionally wrong, and operationally wrong, since it
produces negative forecasts that then have to be clipped away.

**2. Scaled error, not percentage error.**

MAPE is **undefined for 68 % of this dataset**. The usual workaround — dropping zero actuals — does not fix
the metric, it silently changes the question being asked: the surviving observations are systematically the
faster-moving series. This repository reports **RMSSE** and the dollar-weighted **WRMSSE** as primary, and
reports MAPE only to demonstrate that it induces a *different ranking of models*.

```
                 √( (1/h) Σ (y_t − ŷ_t)² )
RMSSE_i  =  ─────────────────────────────────────
             √( (1/(n−1)) Σ (y_t − y_{t−1})² )
```

Note the denominator is a **root-mean-square** of first differences, matching the order of the numerator.
Using a mean *absolute* difference there — a common implementation slip — makes the ratio dimensionally
inconsistent and no longer scale-free.

## Approach

| Component | Choice | Why |
|---|---|---|
| Model | One global gradient-boosted ensemble across all 30,490 series | Individual series are short and noisy; cross-learning trades a little bias for a large variance reduction |
| Objective | Tweedie, `p = 1.1` | Correct likelihood for zero-inflated non-negative counts |
| Intervals | Direct quantile regression at α = 0.025 / 0.975 | No symmetry assumption; the lower bound may be exactly zero |
| Baselines | Naive, seasonal naive (7), SES, **Croston–SBA** | Croston–SBA is the reference method for intermittent demand |
| Validation | Rolling-origin, 3 folds × 28-day horizon | Random k-fold would leak future data across a shared calendar |

**Features** (all built with grouped shifts so nothing from day *t* or later enters row *t*): sales lags
{1,2,3,7,14,21,28}; rolling mean/sd over {7,14,28}; EWM at α ∈ {0.3, 0.1}; calendar (day-of-week, month,
day-of-year, events, state-specific SNAP flags, SNAP × day-of-week); price level, week-on-week change,
ratio to a 28-day mean, and a derived promotion flag; and item/department/category/store/state identity.

## Repository layout

```
├── src/
│   ├── ingestion/       Raw load and database staging
│   ├── etl/             Cleaning and feature engineering
│   ├── models/          ARIMA, Prophet, XGBoost (point + quantile), Isolation Forest
│   ├── evaluation/      Metric suite and walk-forward backtest
│   ├── risk/            Shortfall and volatility analytics
│   ├── serving/         Streamlit dashboard
│   └── deploy/          Hugging Face and Supabase publication
├── dags/                Airflow orchestration
├── db/                  Schema and initialisation
├── tests/               Unit tests for ETL, metrics and models
└── report/              Technical paper (LaTeX source + PDF)
```

## Quick start

```bash
git clone https://github.com/rudraakshreddy/demand-sentinel.git
cd demand-sentinel
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make up            # Postgres + Airflow + pgAdmin via docker compose
make pipeline      # ingest → clean → features → train → evaluate
make serve         # Streamlit dashboard on :8501
```

The M5 source files are not redistributed here; place `calendar.csv`, `sales_train_evaluation.csv` and
`sell_prices.csv` in `data/raw/`, or pull the processed panel from the Hugging Face dataset linked above.

## Scope

**In scope.** Daily store–item forecasting at a 28-day horizon, point forecasts and 95 % prediction
intervals, scaled-error and dollar-weighted evaluation, interval calibration and sharpness.

**Out of scope.** Hierarchical reconciliation (forecasts are bottom-level only and are not made coherent
with store/department/state aggregates); deep sequence models; the mapping from forecast distribution to
order quantity or safety stock; price and promotion *optimisation* (prices are exogenous covariates, and no
elasticity or causal claim is made); cross-product cannibalisation; new-product cold start; and
generalisation beyond US grocery retail.

## Limitations

- **One dataset, one retailer.** M5 covers three states and ten stores of a single chain over 2011–2016.
- **Training window.** The global model trains on a trailing 730-day window with row subsampling for
  tractability; all 30,490 series remain represented, but the model does not see the full history.
- **Bottom level only.** No hierarchical coherence is enforced.
- **Fixed hyperparameters.** Tuned once rather than per fold, so reported accuracy is not the maximum
  attainable.
- **Point-in-time prices.** Weekly prices are treated as known over the horizon, which is realistic for
  planned promotions but optimistic for reactive repricing.

## Citation

```bibtex
@techreport{reddy2026demandsentinel,
  title  = {Demand Sentinel: Global Gradient-Boosted Forecasting for Intermittent Retail Demand},
  author = {Yeddula Rudraaksh Reddy and S. N. Chakri},
  year   = {2026},
  type   = {Technical Report},
  url    = {https://github.com/rudraakshreddy/demand-sentinel}
}
```

## Authors

- **Yeddula Rudraaksh Reddy** — primary and corresponding author ·
  [yeddularudraaksh@gmail.com](mailto:yeddularudraaksh@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/rudraakshreddy) · [GitHub](https://github.com/rudraakshreddy)
- **S. N. Chakri** — [snchakrim@gmail.com](mailto:snchakrim@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/snchakri) · [GitHub](https://github.com/snchakri) ·
  [snchakri.com](https://snchakri.com)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The M5 dataset is the property of its publishers and is not redistributed by this repository.
