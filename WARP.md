# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.
``

Project: Customer_Churn — MLOps system for real-time customer churn prediction with FastAPI, MLflow, Airflow, Prometheus/Grafana, and Postgres.

Core commands
- Install deps (local):
  - python -m pip install --upgrade pip && pip install -r requirements.txt
- Run tests (all / single / by expression):
  - pytest
  - pytest tests/test_retrain.py::TestTrainCombined::test_train_combined_empty_production_returns_minus_one -q
  - pytest -k "monitoring and not slow" -q
- Linting: none configured in this repo (CI only runs pytest).
- Bring up full stack (recommended for dev):
  - cp .env.example .env
  - docker compose up --build -d
  - Key URLs: API http://localhost:8000, Airflow http://localhost:8080 (admin/admin), MLflow http://localhost:5001, Grafana http://localhost:3000 (admin/admin)
- Rebuild/stop:
  - docker compose build
  - docker compose down
- Run the API only (compose):
  - docker compose up -d fastapi
- Run the API locally (SQLite, minimal):
  - ENV=test uvicorn src.api.main:app --reload --port 8000
- Database migrations (Alembic):
  - DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/churn_db" alembic upgrade head
- Initial admin user (created on API start via seed):
  - email: admin@example.com, password: admin
- Train and log models to MLflow:
  - python -m src.training.train
- Retrain (features-only or combining production data):
  - python -m src.training.retrain --mode features
  - python -m src.training.retrain --mode combined
- Generate monitoring reports locally:
  - python scripts/generate_reports.py --baseline data/features/features.csv --production data/production/production.csv
  - python scripts/generate_reports.py --quality-only data/production/production.csv
- Verify monitoring setup:
  - python scripts/verify_monitoring_setup.py

Architecture at a glance
- Orchestration and services (docker-compose.yml):
  - fastapi: Prediction API and admin endpoints (uvicorn), talks to Postgres (db) and MLflow registry; syncs data via DVC on startup.
  - db, mlflow_db: Postgres databases for app and MLflow backend store.
  - mlflow: MLflow tracking and model registry, artifacts stored under ./mlflow_artifacts.
  - airflow: Schedules feature building, training, drift detection, and retraining.
  - prometheus + grafana: Metrics scraping and dashboards (Prometheus scrapes API /metrics; alerts defined in prometheus/alerts.yml).
- API layer (src/api):
  - src/api/main.py: FastAPI app with lifespan tasks: init DB (unless ENV=test), dvc pull, preload MLflow models, periodic reload. Exposes:
    - /predict: batch predictions using MLflow pyfunc model.
    - /survey/submit: single prediction with A/B split between Prod/Staging sklearn models; logs exposure events to data/experiments/ab_exposures.csv; appends labeled requests to data/production/production.csv and schedules DVC push.
    - /model/version, /models: model registry info; /monitoring/* for baseline management; /metrics for Prometheus via Instrumentator.
  - Security: API key via header X-API-Key, validated against API_KEY_SECRET (src/api/core/security.py). Some endpoints require Depends(verify_api_key).
  - Persistence: SQLModel models (src/api/entities), Postgres by default; Alembic migrations under migrations/.
- Experimentation and A/B (src/experiments/ab.py):
  - Sticky bucketing using configurable header; deterministic hashing with adjustable B bucket percentage; exposure logging to CSV.
- Data/ETL and training (src/etl, src/training):
  - Preprocessing builds features (encoding, scaling, SMOTE) and writes /opt/airflow/data/{preprocessed,features}/ in container; returns splits for training.
  - Training (src/training/train.py) fits multiple sklearn models, logs metrics/artifacts to MLflow, and registers the best to the registry name CustomerChurnModel_*; helper promotes best to Production or Staging.
  - Retraining (src/training/retrain.py) supports features-only and combined (features + production) modes; logs to experiment Customer_Churn_Retraining and registers to Staging by default.
- Monitoring (src/monitoring):
  - drift.py: Numeric drift via PSI with robust handling of empty/missing production; writes JSON report.
  - reports.py: Evidently-based drift and data-quality HTML/JSON reports; summary report with alert list.
- Airflow DAGs (dags/):
  - etl_train_dag.py: preprocess -> train -> evaluate -> completion (hourly by default in file).
  - drift_retrain_dag.py: build_features -> detect_drift -> generate_reports -> branch (retrain_combined or skip) -> done; uses MLflow at MLFLOW_URI.
- Observability:
  - Prometheus scrapes fastapi:8000/metrics; alerts cover latency, error rate, drift, and accuracy.
  - Grafana provisioning under grafana/provisioning with a default dashboard.

Grafana dashboard (quick start)
- Panels to add (PromQL shown per panel):
  - p95 prediction latency (5m):
    - histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))
  - Prediction error rate (5m):
    - sum(rate(prediction_errors_total[5m])) / sum(rate(prediction_requests_total[5m]))
  - Request volume (req/s):
    - sum(rate(prediction_requests_total[5m]))
  - Online accuracy:
    - model_accuracy
  - Drift overview (last 30m, max across features):
    - max_over_time(feature_drift_statistic[30m])
  - Drift by feature (table):
    - feature_drift_statistic{feature!=""}

- Alerts already configured in prometheus/alerts.yml (thresholds):
  - HighPredictionLatencyP95 > 0.5s for 5m
  - HighPredictionErrorRate > 5% for 10m
  - DataDriftDetected: feature_drift_statistic > 0.2 for 15m
  - LowModelAccuracy < 0.8 for 10m

Environment and configuration
- .env.example keys used across services:
  - DATABASE_URL, API_KEY_SECRET, MLFLOW_URI, ENV (dev/test)
- API runtime env (src/api/main.py):
  - MLFLOW_TRACKING_URI (default http://mlflow:5000), MODEL_REGISTRY_NAME (default CustomerChurnModel), MODEL_STAGE (default Production)
  - A/B: AB_ENABLED, AB_BUCKET_B_PCT, AB_STICKY_HEADER
- Startup behavior:
  - API executes dvc pull -v; survey submissions trigger background dvc push. Ensure DVC remote is configured; compose mounts .dvc and related folders into containers.

Testing notes
- Test runner: pytest (see .github/workflows/ci.yml). CI sets up Python 3.11, installs requirements.txt, then runs pytest.
- The test suite sets ENV=test and overrides DB with in-memory SQLite; Airflow/DAG tests are skipped if Airflow imports are unavailable.
- Typical commands: pytest -q, pytest tests/..., or pytest -k "expr".

Access and credentials (from README.md)
- After docker compose up:
  - API: http://localhost:8000 (default admin created on startup)
    - email: admin@example.com, password: admin
    - Protected endpoints require header: X-API-Key: <API_KEY_SECRET>
  - Airflow: http://localhost:8080 (admin/admin)
  - MLflow: http://localhost:5001
  - Grafana: http://localhost:3000 (admin/admin)
