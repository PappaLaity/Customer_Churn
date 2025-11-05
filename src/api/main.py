import os
import asyncio
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pydantic import BaseModel

from src.api.core.database import init_db
from src.api.core.security import verify_api_key
from src.api.entities.customerInput import InputCustomer
from src.api.routes import auth, users


ENV = os.getenv("ENV", "dev")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):

    if ENV != "test":
        init_db()

    try:
        print("Pulling DVC data...")
        subprocess.run(["dvc", "pull", "-v"], check=True)
        print("DVC data synchronized.")
    except Exception as e:
        print(f"DVC pull failed: {e}")

    app.state.model_A = None
    app.state.model_B = None
    app.state.stag_version = None
    app.state.stag_source = None
    app.state.prod_version = None
    app.state.prod_source = None

    # initial load of sklearn models for A/B
    load_model(MODEL_NAME)
    task = asyncio.create_task(model_reloader(interval=300))

    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Customer Churn Prediction", lifespan=lifespan)

# Expose default HTTP metrics at /metrics
Instrumentator().instrument(app).expose(app)

mlflow.set_experiment("Production_Customer_Churn_API")

# --- MLflow pyfunc model for batch predictions ---
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


@app.get("/model/version", dependencies=[Depends(verify_api_key)])
async def get_model_version():
    return {
        "production_model_version": app.state.prod_version,
        "staging_model_version": app.state.stag_version,
    }


@app.get("/models")
async def get_models():
    models = mlflow.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'", max_results=1000
    )
    return {
        "models": [
            {
                "version": m.version,
                "current_stage": m.current_stage,
                "creation_timestamp": m.creation_timestamp,
                "last_updated_timestamp": m.last_updated_timestamp,
                "source": m.source,
                "run_id": m.run_id,
            }
            for m in models
        ]
    }


@app.get("/customers/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    file_path = Path("data/production/production.csv")
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return JSONResponse(
                    content={"status": "success", "data": [], "count": 0}
                )

            df_reversed = df.iloc[::-1].reset_index(drop=True)
            data = df_reversed.to_dict(orient="records")
            return JSONResponse(
                content={
                    "status": "success",
                    "columns": df_reversed.columns.tolist(),
                    "data": data,
                    "count": len(df_reversed),
                }
            )
        except pd.errors.EmptyDataError:
            return JSONResponse(
                content={
                    "status": "success",
                    "data": [],
                    "count": 0,
                    "message": "Production data file is empty (no headers)",
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error reading production data: {str(e)}"
            )
    else:
        return JSONResponse(
            content={
                "status": "success",
                "data": [],
                "count": 0,
                "message": "No production data found",
            }
        )


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
        PREDICTION_ERRORS.labels(
            model_version=model_version, error_type="bad_input"
        ).inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Optionally separate labels
    y_true = None
    if payload.label_key and payload.label_key in df.columns:
        y_true = df[payload.label_key].to_numpy()
        df = df.drop(columns=[payload.label_key])

    start = time.time()
    try:
        preds = _model.predict(df)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        duration = time.time() - start
        PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)
        PREDICTION_REQUESTS.labels(model_version=model_version).inc()
    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_version=model_version, error_type="inference"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Drift computation (numeric only, if baseline present)
    if _baseline_numeric_sorted:
        try:
            num_df = df.select_dtypes(include=[np.number])
            for col in num_df.columns:
                FEATURE_MEAN.labels(feature=col).set(
                    float(np.nanmean(num_df[col].to_numpy()))
                )
                if col in _baseline_numeric_sorted:
                    sample_sorted = np.sort(num_df[col].to_numpy(dtype=float))
                    if sample_sorted.size > 0:
                        d = _ks_d_stat(_baseline_numeric_sorted[col], sample_sorted)
                        FEATURE_DRIFT_STAT.labels(feature=col).set(float(d))
        except Exception:
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


@app.post("/survey/submit")
async def submit_survey(input: InputCustomer, background_tasks: BackgroundTasks):
    # Predict single record using A/B models if available, else pyfunc model
    start = time.time()
    result = await predict_single(input)
    duration = time.time() - start

    # Log metrics
    PREDICTION_LATENCY.labels(
        model_version=str(app.state.prod_version or _model_version)
    ).observe(duration)
    PREDICTION_REQUESTS.labels(
        model_version=str(app.state.prod_version or _model_version)
    ).inc()

    # Append to production CSV
    file_path = Path("data/production/production.csv")
    # Data Validation
    data = input
    result = await predict_churn(data)
    # Make Prediction
    customer_data = input.model_dump(by_alias=True)
    customer_data["Churn"] = int(result)
    df = pd.DataFrame([customer_data])
    df["Churn"] = result

    csv_columns = [
        "tenure",
        "InternetService_Fiber_optic",
        "Contract_Two_year",
        "PaymentMethod_Electronic_check",
        "No_internet_service",
        "TotalCharges",
        "MonthlyCharges",
        "PaperlessBilling",
        "Churn",
    ]

    # Vérifier si le fichier existe ET non vide
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=csv_columns)
    else:
        df_existing = pd.DataFrame(columns=csv_columns)

    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv(file_path, index=False)
    # Store it in the production data
    background_tasks.add_task(dvc_push_background)
    return {"success": "Thanky you for your submission"}


async def predict_single(data: InputCustomer) -> Dict[str, Any]:
    df = pd.DataFrame([data.model_dump()])
    model_choice = "A"
    if (
        getattr(app.state, "model_A", None) is not None
        and getattr(app.state, "model_B", None) is not None
    ):
        model_choice = "A" if random.random() < 0.8 else "B"

    start = time.time()
    if model_choice == "A" and app.state.model_A is not None:
        preds = app.state.model_A.predict(df)
        model_used = f"Production(v{app.state.prod_version})"
    elif model_choice == "B" and app.state.model_B is not None:
        preds = app.state.model_B.predict(df)
        model_used = f"Staging(v{app.state.stag_version})"
    else:
        _ensure_model_loaded()
        preds = _model.predict(df)
        model_used = f"Registry({MODEL_STAGE})"
    latency = time.time() - start

    return {"model": model_used, "prediction": int(preds[0]), "latency": latency}


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


# # Métriques personnalisées
prediction_counter = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["model_version", "prediction_result"],
)

prediction_duration = Histogram(
    "churn_prediction_duration_seconds", "Time spent processing prediction"
)

active_users = Gauge("churn_api_active_users", "Number of active users")


# # Exemple d'utilisation dans vos endpoints
async def predict_churn(data=None):
    start_time = time.time()

    results = await predict(data)
    result = results["prediction"]
    # Enregistrer les métriques
    prediction_counter.labels(
        model_version="v1.0", prediction_result="churn" if result == 1 else "no_churn"
    ).inc()

    prediction_duration.observe(time.time() - start_time)

    return result


async def predict(data: InputCustomer):

    df = pd.DataFrame([data.model_dump()])
    if app.state.model_A and app.state.model_B:
        model_choice = "A" if random.random() < 0.8 else "B"
    else:
        model_choice = "A"

    start = time.time()
    preds = (
        app.state.model_A.predict(df)
        if model_choice == "A"
        else app.state.model_B.predict(df)
    )
    # result = model_A.predict(df)[0]
    print(f"Predicted result: {preds[0]}")
    latency = time.time() - start

    # Log locally or send to MLflow for analysis
    mlflow.log_metric("latency", latency)
    mlflow.log_param("model_used", model_choice)

    return {
        "model": "Production" if model_choice == "A" else "Staging",
        "prediction": preds[0],
        "latency": latency,
    }


def load_model(model_name="CustomerChurnModel"):

    models = mlflow.search_model_versions(
        filter_string=f"name='{model_name}'", max_results=1000
    )
    for m in models:
        if m.current_stage == "Production":
            app.state.prod_version = m.version
            app.state.prod_source = m.source
        if m.current_stage == "Staging":
            app.state.stag_version = m.version
            app.state.stag_source = m.source

    print(
        f"Production model version: {app.state.prod_version}, source: {app.state.prod_source}"
    )
    print(
        f"Staging model version: {app.state.stag_version}, source: {app.state.stag_source}"
    )
    try:
        if app.state.prod_source:
            app.state.model_A = mlflow.sklearn.load_model(app.state.prod_source)
            print(f"Loaded Production model: {app.state.prod_source}")
        if app.state.stag_source:
            app.state.model_B = mlflow.sklearn.load_model(app.state.stag_source)
            print(f"Loaded Staging model: {app.state.stag_source}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


async def model_reloader(interval: int = 300):
    await asyncio.sleep(5)
    while True:
        try:
            load_model(MODEL_NAME)
        except Exception as e:
            print(f"Erreur lors du rechargement périodique: {e}")
        await asyncio.sleep(interval)


async def dvc_push_background():
    process = await asyncio.create_subprocess_exec(
        "dvc",
        "push",
        "-v",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print("[DVC] Push successful:\n", stdout.decode())
    else:
        print("[DVC] Push failed:\n", stderr.decode())
