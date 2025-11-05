# WARP.md

This file provides guidance when working with code in this repository.

## Common commands

- Environment setup (local):
  - python -m venv .venv && source .venv/bin/activate
  - pip install -r requirements.txt
  - cp .env.example .env and adjust variables (DATABASE_URL, IP_ADDRESS, API_KEY_SECRET)
- Run API locally:
  - uvicorn src.api.main:app --reload --port 8000
- Run the full stack (DB, API, MLflow, Airflow, Prometheus, Grafana):
  - docker compose up --build -d
  - Logs for a service: docker compose logs -f fastapi (or db, mlflow, airflow, prometheus, grafana)
- Database migrations (Alem bic):
  - Create revision: DATABASE_URL=<db_url> alembic revision --autogenerate -m "message"
  - Apply latest: DATABASE_URL=<db_url> alembic upgrade head
- Tests:
  - pytest
  - Run a single test: pytest tests/api/test_api.py::test_root -q
  - Filter by pattern: pytest -k "health"
- Lint/format: No project-wide linter configured. If desired, use ruff/black locally.

## Architecture overview

- API (FastAPI, src/api)
  - Entrypoint: src/api/main.py creates the FastAPI app, wires routers, and exposes Prometheus metrics via prometheus-fastapi-instrumentator at /metrics.
  - Auth and Users: src/api/routes/auth.py and src/api/routes/users.py implement registration, login, and CRUD over a SQLModel-backed User (src/api/entities/users.py). API key protection via src/api/core/security.py (X-API-Key header).
  - Persistence: src/api/core/database.py configures SQLModel engine. Alembic migrations live in migrations/, configured by alembic.ini and migrations/env.py. A seed script exists at src/api/core/seed.py.
- Data/ML pipeline (src/etl, src/training)
  - ETL: src/etl/extract.py loads the Telco churn CSV; src/etl/preprocessing.py encodes, selects features, splits, and persists encoders to encoders.pkl.
  - Training: src/training/train.py trains multiple sklearn models, logs metrics/artifacts to MLflow, and registers/promotes the best model in the registry (CustomerChurnModel). Tracking URI comes from IP_ADDRESS (port 5001). Local artifact store: mlruns/.
- Orchestration (Airflow, dags/)
  - dags/etl_train_dag.py runs preprocessing → training → evaluation inside the Airflow container. Airflow image is built via Dockerfile.airflow and installs requirements-airflow.txt.
- Infrastructure (docker-compose.yml)
  - Services: Postgres (app DB), Postgres (MLflow DB), FastAPI (Dockerfile.api runs alembic upgrade, seeds, then uvicorn), MLflow server (Dockerfile.mlflow), Airflow, Prometheus, Grafana.
  - Prometheus scrapes FastAPI at /metrics. Grafana mounts provisioning/ for datasources/dashboards.
- CI/CD
  - .github/workflows/ci.yml: Python 3.11, installs requirements, runs pytest on develop.
  - .github/workflows/deploy.yml: On push to main, builds Compose and deploys to Azure VM over SSH, running docker compose up -d and pruning.

## Notes and gotchas

- Tests set ENV="test" to skip DB init on import and to use an in-memory SQLite session override (tests/conftest.py). When running API locally, ensure ENV is not "test".
- The API expects an X-API-Key header for protected routes; set API_KEY_SECRET in the environment (src/api/core/security.py). .env.example shows typical variables.
- MLflow in Compose is exposed on host port 5001 but intra-network access is http://mlflow:5000; training uses IP_ADDRESS to form the external tracking URI.
