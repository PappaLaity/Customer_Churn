# 📊 Customer Churn Prediction System - Full Project Report

**Date**: December 2024  
**Version**: 1.0  
**Status**: Production Ready ✅

---

## 1. Executive Summary

This project is an **end-to-end Machine Learning system** designed to predict customer churn for telecommunications companies. It combines:

- **ML Pipeline**: Automated training, evaluation, and deployment
- **Production API**: Real-time and batch predictions
- **Monitoring**: Drift detection, model performance tracking
- **Web Interface**: Vue.js dashboard for end-users

The system processes customer data and predicts churn probability with **74% recall** (primary metric), enabling proactive customer retention strategies.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CUSTOMER CHURN PREDICTION SYSTEM                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│   │   Frontend   │────▶│   FastAPI    │────▶│  MLflow Registry     │   │
│   │   (Vue.js)   │     │   Backend    │     │  (Model Versions)    │   │
│   └──────────────┘     └──────────────┘     └──────────────────────┘   │
│          │                    │                       │                 │
│          │                    ▼                       │                 │
│          │             ┌──────────────┐               │                 │
│          │             │  PostgreSQL  │               │                 │
│          │             │  (Database)  │               │                 │
│          │             └──────────────┘               │                 │
│          │                    │                       │                 │
│          ▼                    ▼                       ▼                 │
│   ┌──────────────────────────────────────────────────────────────────┐ │
│   │                    MONITORING STACK                              │ │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐ │ │
│   │  │ Prometheus │──│  Grafana   │  │  Evidently + Alibi Detect  │ │ │
│   │  └────────────┘  └────────────┘  └────────────────────────────┘ │ │
│   └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐ │
│   │                    ML PIPELINE (AIRFLOW)                         │ │
│   │  Data Ingestion → Preprocessing → Training → Evaluation → Deploy │ │
│   └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Vue.js 3, Vite | User interface |
| **Backend** | FastAPI, Python 3.11 | REST API |
| **ML Framework** | Scikit-learn, XGBoost | Model training |
| **MLOps** | MLflow, DVC | Model registry, data versioning |
| **Orchestration** | Apache Airflow | Pipeline automation |
| **Database** | PostgreSQL | Data persistence |
| **Monitoring** | Prometheus, Grafana | Metrics & dashboards |
| **Drift Detection** | Evidently AI, Alibi Detect | Data/model drift |
| **Infrastructure** | Docker, Docker Compose | Containerization |

### 2.3 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `fastapi` | 8000 | Main API server |
| `frontend` | 8081 | Vue.js web application |
| `mlflow` | 5001 | Model registry UI |
| `airflow` | 8080 | Workflow orchestration |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Monitoring dashboards |
| `nginx-reports` | 8888 | Evidently HTML reports |
| `db` | 5432 | PostgreSQL database |

---

## 3. Machine Learning Pipeline

### 3.1 Data Flow

```
Raw Data (CSV) → Preprocessing → Feature Engineering → Model Training → Evaluation → Registry
```

### 3.2 Model Selection

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.75 | 0.72 | 0.73 |
| Random Forest | 0.81 | 0.79 | 0.74 | 0.76 |
| **HistGradientBoosting** | **0.82** | **0.80** | **0.74** | **0.77** |
| Neural Network (MLP) | 0.79 | 0.77 | 0.71 | 0.74 |

**Selected Model**: HistGradientBoosting (best recall for churn detection)

### 3.3 Features Used (8 features)

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | Numeric | Months as customer |
| `MonthlyCharges` | Numeric | Monthly bill amount |
| `TotalCharges` | Numeric | Total amount paid |
| `Contract_Two_year` | Boolean | Has 2-year contract |
| `InternetService_Fiber_optic` | Boolean | Fiber internet subscriber |
| `PaymentMethod_Electronic_check` | Boolean | Pays via e-check |
| `No_internet_service` | Integer | No internet (0/1) |
| `PaperlessBilling` | Integer | Paperless billing (0/1) |

### 3.4 Airflow DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `Customer_Churn_DVC_pipeline_Train_Eval` | Manual | Full training pipeline |
| `drift_retrain_dag` | Daily | Drift detection & auto-retraining |

---

## 4. API Endpoints

### 4.1 Prediction Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/survey/submit` | Public | Single prediction (rate-limited) |
| POST | `/predict` | API Key | Batch predictions |
| POST | `/predict/csv` | API Key | CSV file predictions |
| GET | `/predict/batch/{id}` | API Key | Retrieve batch results |

### 4.2 Monitoring Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Prometheus metrics |
| GET | `/metrics/evidently` | Evidently drift metrics |
| GET | `/metrics/alibi` | Alibi Detect metrics |
| GET | `/metrics/mlflow` | Model performance metrics |
| GET | `/monitoring/evidently/status` | Evidently status JSON |
| GET | `/monitoring/alibi/status` | Alibi status JSON |
| GET | `/monitoring/mlflow/status` | MLflow metrics JSON |

### 4.3 A/B Testing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/ab/config` | Current A/B config |
| PUT | `/ab/config` | Update A/B settings |
| GET | `/ab/results` | Statistical analysis |

---

## 5. Monitoring & Observability

### 5.1 Grafana Dashboard

**URL**: `http://localhost:3000/d/churn-monitoring/`

#### Panels Available

| Panel | Metric | Source |
|-------|--------|--------|
| 🚨 Evidently Drift | `evidently_drift_detected` | Evidently |
| Drift Share % | `evidently_drift_share` | Evidently |
| Drifted Features | `evidently_drifted_features_count` | Evidently |
| 🎯 Model Recall | `model_recall{stage="Production"}` | MLflow |
| 🧪 Alibi Drift Status | `alibi_drift_detected` | Alibi Detect |
| Numerical Drift | `alibi_numerical_drift_detected` | Alibi Detect |
| Categorical Drift | `alibi_categorical_drift_detected` | Alibi Detect |
| Feature P-Values | `alibi_feature_pvalue` | Alibi Detect |
| Prediction Latency | `prediction_latency_seconds` | FastAPI |
| Throughput | `prediction_requests_total` | FastAPI |
| Error Rate | `prediction_errors_total` | FastAPI |

### 5.2 Drift Detection Methods

| Tool | Method | Use Case |
|------|--------|----------|
| **Evidently AI** | Distribution comparison | Visual reports, quick overview |
| **Alibi Detect** | KS Test (numeric), Chi² (categorical) | Statistical significance (p-value) |

### 5.3 Current Drift Status

Based on Alibi Detect analysis:

| Feature | P-Value | Drift? |
|---------|---------|--------|
| MonthlyCharges | 2.14e-21 | ⚠️ YES |
| tenure | 5.07e-16 | ⚠️ YES |
| TotalCharges | 4.44e-14 | ⚠️ YES |
| Contract_Two_year | 0.0017 | ⚠️ YES |
| No_internet_service | 0.003 | ⚠️ YES |

---

## 6. Security

### 6.1 Authentication

| Endpoint Type | Auth Method |
|--------------|-------------|
| Public (`/survey/submit`) | Rate limiting (10/min) |
| Protected (`/predict`, `/ab/*`) | API Key header (`X-API-Key`) |
| Admin | JWT token (future) |

### 6.2 Rate Limiting

- **Public endpoints**: 10 requests/minute per IP
- **Protected endpoints**: Unlimited (API key required)

### 6.3 CORS Configuration

```python
origins = [
    "http://localhost:8081",  # Frontend dev
    "http://localhost:3000",  # Grafana
]
```

---

## 7. Data Management

### 7.1 DVC (Data Version Control)

- **Remote**: Local filesystem (`remote/`)
- **Tracked Data**: `data/` directory
- **Auto-sync**: On API startup

### 7.2 Data Paths

| Path | Content |
|------|---------|
| `data/input/` | Raw Telco dataset |
| `data/features/` | Preprocessed features |
| `data/production/` | Production predictions |
| `data/monitoring/reports/` | Evidently reports |
| `drifts/monitoring/` | Alibi drift reports |

---

## 8. Deployment

### 8.1 Quick Start

```bash
# Clone and start
git clone <repo>
cd Customer_Churn
docker-compose up -d

# Access services
open http://localhost:8000/docs      # API Swagger
open http://localhost:8081           # Frontend
open http://localhost:3000           # Grafana
open http://localhost:5001           # MLflow
open http://localhost:8080           # Airflow
```

### 8.2 Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@db:5432/churn

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_REGISTRY_NAME=CustomerChurnModel
MODEL_STAGE=Production

# Security
API_KEY_SECRET=<your-secret-key>

# A/B Testing
AB_ENABLED=true
```

---

## 9. Key Metrics Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Model Recall** | 74% | ✅ Good |
| **Model Accuracy** | 82% | ✅ Good |
| **API Latency (p95)** | <500ms | ✅ Good |
| **Drift Detected** | 5 features | ⚠️ Review |
| **Services Running** | 8/8 | ✅ Healthy |

---

## 10. Future Roadmap

- [ ] **Kubernetes Deployment**: Scale to K8s for high availability
- [ ] **SHAP Explainability**: Feature importance in predictions
- [ ] **Shadow Mode**: Test new models without affecting production
- [ ] **Email Alerts**: Automated notifications on drift/issues
- [ ] **Model A/B Testing**: Statistical comparison of model versions

---

## 11. Documentation Index

| Document | Description |
|----------|-------------|
| [MONITORING.md](./MONITORING.md) | Monitoring setup guide |
| [API_SECURITY.md](./API_SECURITY.md) | Security implementation |
| [AIRFLOW_ARCHITECTURE.md](./AIRFLOW_ARCHITECTURE.md) | Pipeline design |
| [ACCESS_CONTROL.md](./ACCESS_CONTROL.md) | Role-based access |
| [CICD_SETUP.md](./CICD_SETUP.md) | CI/CD configuration |

---

**Report Generated**: December 5, 2024  
**System Status**: ✅ Production Ready
