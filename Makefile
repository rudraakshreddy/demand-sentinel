.PHONY: help setup ingest etl train risk evaluate serve airflow-up airflow-down test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python dependencies
	pip install -r requirements.txt

airflow-up:  ## Start PostgreSQL + Airflow via Docker Compose
	docker compose up -d

	@echo "Airflow UI: http://localhost:8080 (airflow/airflow)"
	@echo "pgAdmin:    http://localhost:5050 (admin@dfp.local/admin)"

airflow-down:  ## Stop all containers
	docker compose down

ingest:  ## Copy raw M5 data into data/raw/ and load into PostgreSQL
	python src/ingestion/copy_raw.py
	python src/ingestion/load_db.py

etl:  ## Run full ETL pipeline (clean + feature engineering)
	python src/etl/pipeline.py

train:  ## Train all models (ARIMA, Prophet, XGBoost, IsolationForest)
	python src/models/arima_model.py
	python src/models/prophet_model.py
	python src/models/xgboost_model.py
	python src/models/isolation_forest.py

risk:  ## Compute risk layer (volatility + shortfall)
	python src/risk/volatility.py
	python src/risk/shortfall.py

evaluate:  ## Run backtest evaluation and save metrics
	python src/evaluation/backtest.py

serve:  ## Launch Streamlit dashboard
	streamlit run src/serving/dashboard.py --server.port 8501

test:  ## Run unit tests
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

report:  ## Generate LaTeX tables from pipeline outputs and compile PDF
	python report/generate_report.py
	cd report && pdflatex -interaction=nonstopmode main.tex
	cd report && pdflatex -interaction=nonstopmode main.tex
	@echo "PDF compiled: report/main.pdf"

git-init:  ## Initialise Git repo and make first commit
	git init
	echo "data/raw/" >> .gitignore
	echo "data/processed/" >> .gitignore
	echo "logs/" >> .gitignore
	echo "__pycache__/" >> .gitignore
	echo "*.pyc" >> .gitignore
	echo ".env" >> .gitignore
	git add .
	git commit -m "feat: initial commit — 7-layer demand forecasting platform"

clean:  ## Remove all generated data and model artifacts
	rm -rf data/processed/* data/external/* logs/*.log
