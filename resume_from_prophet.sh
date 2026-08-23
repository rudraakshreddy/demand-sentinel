#!/bin/bash
# Resume from Prophet — SARIMA results already on disk

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dfp

PROJ="$HOME/demand-forecasting-platform"
cd "$PROJ"

export PYTHONPATH="$PROJ"
export DB_HOST=localhost DB_PORT=5432 DB_USER=forecast_user DB_PASS=forecast_pass DB_NAME=demand_forecasting
export FORECAST_HORIZON=28 N_CV_FOLDS=5 OPTUNA_TRIALS=50 RANDOM_SEED=42 XGB_DEVICE=cuda

mkdir -p logs data/processed data/processed/models

SEP="══════════════════════════════════════════════════"
step() { echo ""; echo "$SEP"; echo "  STAGE: $1  [$(date '+%H:%M:%S')]"; echo "$SEP"; }

step "MODEL: Prophet"
time python3 src/models/prophet_model.py 2>&1 | tee logs/prophet.log || echo "WARN: prophet exited non-zero"

step "MODEL: XGBoost (RTX 5090 CUDA)"
time python3 src/models/xgboost_model.py 2>&1 | tee logs/xgboost.log || echo "WARN: xgboost exited non-zero"

step "MODEL: Isolation Forest"
time python3 src/models/isolation_forest.py 2>&1 | tee logs/isolation_forest.log || echo "WARN: isolation_forest exited non-zero"

step "RISK: Volatility"
time python3 src/risk/volatility.py 2>&1 | tee logs/volatility.log || echo "WARN: volatility exited non-zero"

step "RISK: Shortfall"
time python3 src/risk/shortfall.py 2>&1 | tee logs/shortfall.log || echo "WARN: shortfall exited non-zero"

step "EVAL: Backtest"
time python3 src/evaluation/backtest.py 2>&1 | tee logs/backtest.log || echo "WARN: backtest exited non-zero"

step "TESTS: pytest"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee logs/tests.log || true

step "REPORT: generate LaTeX tables"
python3 report/generate_report.py 2>&1 | tee logs/report_gen.log || echo "WARN: report gen exited non-zero"

step "REPORT: pdflatex (2 passes)"
cd report
pdflatex -interaction=nonstopmode main.tex > /tmp/latex1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/latex2.log 2>&1
ls -lh main.pdf 2>/dev/null && echo "PDF OK" || echo "PDF failed — check /tmp/latex1.log"
cd ..

echo ""
echo "PIPELINE COMPLETE — $(date)"
echo ""
ls -lh data/processed/*.parquet data/processed/*.csv 2>/dev/null || true
ls -lh data/processed/models/ 2>/dev/null || true
ls -lh report/main.pdf 2>/dev/null || true
