# import os
# import random
# from fastapi import Depends, FastAPI, HTTPException, requests
# from fastapi.responses import JSONResponse
# from fastapi.security import HTTPBearer
# from fastapi.openapi.utils import get_openapi
# import mlflow
# import pandas as pd
# from prometheus_fastapi_instrumentator import Instrumentator
# from src.api.entities.customerInput import InputCustomer
# from src.api.core.security import verify_api_key
# from src.api.routes import users, auth
# from src.api.core.database import init_db
# from pathlib import Path
# from pydantic import BaseModel

# from prometheus_client import Counter, Histogram, Gauge
# import time


# app = FastAPI(title="Customer Churn Prediction")

# Instrumentator().instrument(app).expose(app)


# mlflow.set_tracking_uri("http://mlflow:5000")

# # model_A = mlflow.pyfunc.load_model("models:/CustomerChurnModel/Production")

# churn = ["No", "Yes"]
# # Only initialize the database on app import when not running tests.
# # Tests set ENV="test" in `tests/conftest.py` before importing the app.
# ENV = os.getenv("ENV", "dev")
# if ENV != "test":
#     init_db()


# @app.get("/")
# async def home():
#     return {"msg": "Customer Churn System"}


# @app.get("/customers/infos", dependencies=[Depends(verify_api_key)])
# async def get_customers_infos():
#     infos = []
#     file_path = Path("Data/production/customer_production_data.csv")
#     if file_path.exists():
#         try:
#             df = pd.read_csv(file_path)
#             if df.empty or len(df) == 0:
#                 return JSONResponse(
#                     content={"status": "success", "data": [], "count": 0}
#                 )

#             df_reversed = df.iloc[::-1].reset_index(drop=True)
#             return JSONResponse(
#                 content={
#                     "status": "success",
#                     "columns": df_reversed.columns.tolist(),
#                     "data": df_reversed.values.tolist(),
#                     "count": len(df_reversed),
#                 }
#             )
#         except pd.errors.EmptyDataError:
#             # Fichier complètement vide (pas même d'en-têtes)
#             return JSONResponse(
#                 content={
#                     "status": "success",
#                     "data": [],
#                     "count": 0,
#                     "message": "Production data file is empty (no headers)",
#                 }
#             )
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Error reading production data: {str(e)}"
#             )
#     else:
#         return JSONResponse(
#             content={
#                 "status": "success",
#                 "data": [],
#                 "count": 0,
#                 "message": "No production data found",
#             }
#         )


# @app.get("/health", dependencies=[Depends(verify_api_key)])
# async def check_healh():
#     return {"check": "I'm ok! No worry"}


# @app.post("/survey/submit")
# async def submit_survey(input: InputCustomer = None):

#     # Data Validation
#     data = input
#     # Prepare Data for Prediction
#     # Make Prediction
#     # Prepare Input and Prediction
#     prediction = predict_churn(data)
#     # Store it in the production data

#     return {"success": "Thanky you for your submission"}


# app.include_router(users.router)
# app.include_router(auth.router)


# # # Métriques personnalisées
# prediction_counter = Counter(
#     "churn_predictions_total",
#     "Total number of churn predictions",
#     ["model_version", "prediction_result"],
# )

# prediction_duration = Histogram(
#     "churn_prediction_duration_seconds", "Time spent processing prediction"
# )

# active_users = Gauge("churn_api_active_users", "Number of active users")


# # # Exemple d'utilisation dans vos endpoints
# async def predict_churn(data=None):
#     start_time = time.time()

#     # Votre logique de prédiction
#     # result = your_model.predict(data)
#     result = random.randint(0, 1)
#     # result = predict(data)
#     time.sleep(5)
#     # Enregistrer les métriques
#     prediction_counter.labels(
#         model_version="v1.0", prediction_result="churn" if result == 1 else "no_churn"
#     ).inc()

#     prediction_duration.observe(time.time() - start_time)

#     return result


# # @app.post("/predict")
# async def predict(data: dict):
#     df = pd.DataFrame([data])
#     model_choice = "A" if random.random() < 0.8 else "B"

#     start = time.time()
#     # preds = model_A.predict(df) if model_choice == "A" else model_B.predict(df)
#     latency = time.time() - start

#     # Log locally or send to MLflow for analysis
#     mlflow.log_metric("latency", latency)
#     mlflow.log_param("model_used", model_choice)

#     return {
#         "model": "Production" if model_choice == "A" else "Staging",
#         # "prediction": preds.tolist(),
#         "latency": latency,
#     }




import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.api.core.database import init_db
from src.api.core.security import verify_api_key
from src.api.routes import auth, users
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction")

# Expose default HTTP metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Only initialize the database on app import when not running tests.
ENV = os.getenv("ENV", "dev")
if ENV != "test":
    init_db()

# --- MLflow model loading ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None
_model_version = MODEL_STAGE


def _ensure_model_loaded():
    global _model, _model_version
    if _model is not None:
        return
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        _model = mlflow.pyfunc.load_model(uri)
        try:
            client = MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                _model_version = versions[0].version
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


# --- Prometheus custom metrics ---
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["model_version"]
)
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total prediction requests", ["model_version"]
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Prediction errors", ["model_version", "error_type"]
)
FEATURE_DRIFT_STAT = Gauge(
    "feature_drift_statistic",
    "KS two-sample D statistic for numeric features (higher=worse)",
    ["feature"],
)
FEATURE_MEAN = Gauge("feature_mean", "Online mean of numeric features", ["feature"])
MODEL_ACCURACY = Gauge("model_accuracy", "Cumulative online accuracy")

# State for accuracy and drift baselines
_total_with_label = 0
_correct_with_label = 0
_baseline_numeric_sorted: Dict[str, np.ndarray] = {}


class PredictPayload(BaseModel):
    instances: List[Dict[str, Any]]
    return_proba: Optional[bool] = False
    label_key: Optional[str] = None


class BaselinePayload(BaseModel):
    numeric: Dict[str, List[float]] = {}


@app.get("/")
async def home():
    return {"msg": "Customer Churn System"}


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def check_health():
    return {"check": "I'm ok! No worry"}


@app.post("/monitoring/baseline", dependencies=[Depends(verify_api_key)])
async def set_baseline(payload: BaselinePayload):
    global _baseline_numeric_sorted
    _baseline_numeric_sorted = {
        f: np.sort(np.asarray(vals, dtype=float)) for f, vals in payload.numeric.items()
    }
    return {"status": "ok", "features": list(_baseline_numeric_sorted.keys())}


@app.get("/monitoring/baseline", dependencies=[Depends(verify_api_key)])
async def get_baseline():
    return {"features": list(_baseline_numeric_sorted.keys())}


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(payload: PredictPayload):
    _ensure_model_loaded()
    model_version = str(_model_version)

    # Build dataframe
    try:
        df = pd.DataFrame(payload.instances)
    except Exception as e:
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="bad_input").inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Optionally separate labels
    y_true = None
    if payload.label_key and payload.label_key in df.columns:
        y_true = df[payload.label_key].to_numpy()
        df = df.drop(columns=[payload.label_key])

    start = time.time()
    try:
        preds = _model.predict(df)
        if hasattr(preds, "tolist"):
            preds_list = preds.tolist()
        else:
            preds_list = list(preds)
        duration = time.time() - start
        PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)
        PREDICTION_REQUESTS.labels(model_version=model_version).inc()
    except Exception as e:
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="inference").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Drift computation (numeric only, if baseline present)
    if _baseline_numeric_sorted:
        try:
            num_df = df.select_dtypes(include=[np.number])
            for col in num_df.columns:
                FEATURE_MEAN.labels(feature=col).set(float(np.nanmean(num_df[col].to_numpy())))
                if col in _baseline_numeric_sorted:
                    sample_sorted = np.sort(num_df[col].to_numpy(dtype=float))
                    if sample_sorted.size > 0:
                        d = _ks_d_stat(_baseline_numeric_sorted[col], sample_sorted)
                        FEATURE_DRIFT_STAT.labels(feature=col).set(float(d))
        except Exception:
            # do not fail predictions if drift calc fails
            pass

    # Online accuracy if label provided
    global _total_with_label, _correct_with_label
    if y_true is not None:
        try:
            correct = np.sum((np.asarray(preds) == np.asarray(y_true)).astype(int))
            _total_with_label += len(preds_list)
            _correct_with_label += int(correct)
            MODEL_ACCURACY.set(_correct_with_label / max(1, _total_with_label))
        except Exception:
            pass

    return {"predictions": preds_list, "model_version": model_version}


def _ks_d_stat(a_sorted: np.ndarray, b_sorted: np.ndarray) -> float:
    """Compute the two-sample KS D statistic given two sorted arrays."""
    a_n = a_sorted.size
    b_n = b_sorted.size
    i = j = 0
    d = 0.0
    while i < a_n and j < b_n:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / a_n - j / b_n))
    # handle tails
    d = max(d, abs(1.0 - j / b_n))
    d = max(d, abs(i / a_n - 1.0))
    return float(d)


# Include other routers
app.include_router(users.router)
app.include_router(auth.router)
