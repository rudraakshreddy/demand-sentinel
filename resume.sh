#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dfp

PROJ="$HOME/demand-forecasting-platform"
cd "$PROJ"

export PYTHONPATH="$PROJ"
export DB_HOST=localhost DB_PORT=5432 DB_USER=forecast_user DB_PASS=forecast_pass DB_NAME=demand_forecasting
export FORECAST_HORIZON=28 N_CV_FOLDS=5 OPTUNA_TRIALS=50 RANDOM_SEED=42 XGB_DEVICE=cuda

SEP="══════════════════════════════════════════════════"
step() { echo ""; echo "$SEP"; echo "  STAGE: $1  [$(date '+%H:%M:%S')]"; echo "$SEP"; }

step "MODEL: XGBoost (Optuna skipped, hardcoded best params)"
time python3 src/models/xgboost_model.py 2>&1 | tee logs/xgboost_resume.log

step "MODEL: Isolation Forest (memory fix applied)"
time python3 src/models/isolation_forest.py 2>&1 | tee logs/isolation_forest_resume.log

step "RISK: Volatility"
time python3 src/risk/volatility.py 2>&1 | tee logs/volatility_resume.log

step "RISK: Shortfall"
time python3 src/risk/shortfall.py 2>&1 | tee logs/shortfall_resume.log

step "EVAL: Backtest"
time python3 src/evaluation/backtest.py 2>&1 | tee logs/backtest_resume.log

step "REPORT: generate LaTeX tables"
python3 report/generate_report.py 2>&1 | tee logs/report_gen.log

step "REPORT: pdflatex (2 passes)"
cd report
pdflatex -interaction=nonstopmode main.tex > /tmp/latex1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/latex2.log 2>&1
cd ..

echo "RESUME COMPLETE"
