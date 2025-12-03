"""
Routes de prédiction
- /predict : Prédictions batch
- /survey/submit : Soumission formulaire client
"""

import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from prometheus_client import Counter, Gauge, Histogram

from src.api.core.config import settings
from src.api.core.logger import api_logger as logger
from src.api.core.security import verify_api_key
from src.api.entities.customerInput import InputCustomer
from src.api.schemas.predict import (
    PredictPayload,
    PredictResponse,
    SurveyResponse,
)
from src.api.services.ml_service import (
    get_model_versions,
    predict_batch,
    predict_single_ab,
)
from src.api.services.monitoring_service import (
    compute_ks_statistic,
    get_baseline_for_feature,
    update_accuracy,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Prometheus metrics
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency", ["model_version"]
)
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total prediction requests", ["model_version"]
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Prediction errors", ["model_version", "error_type"]
)
FEATURE_DRIFT_STAT = Gauge(
    "feature_drift_statistic",
    "KS statistic for numeric features",
    ["feature"],
)
FEATURE_MEAN = Gauge("feature_mean", "Online mean of numeric features", ["feature"])
MODEL_ACCURACY = Gauge("model_accuracy", "Cumulative online accuracy")


@router.post("", dependencies=[Depends(verify_api_key)], response_model=PredictResponse)
async def predict_endpoint(payload: PredictPayload):
    """
    Prédiction batch avec monitoring de drift

    Body:
        - instances: Liste de dicts avec les features
        - label_key: Nom de la colonne contenant les vraies valeurs (optionnel)
        - return_proba: Retourner les probabilités (non implémenté)
    """
    model_version = get_model_versions()["pyfunc_model_version"]

    # Build DataFrame
    try:
        df = pd.DataFrame(payload.instances)
    except Exception as e:
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="bad_input").inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Séparer les labels si fournis
    y_true = None
    if payload.label_key and payload.label_key in df.columns:
        y_true = df[payload.label_key].to_numpy()
        df = df.drop(columns=[payload.label_key])

    # Prédiction
    start = time.time()
    try:
        preds = predict_batch(df)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        duration = time.time() - start

        PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)
        PREDICTION_REQUESTS.labels(model_version=model_version).inc()

    except Exception as e:
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="inference").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Drift computation (numeric only)
    try:
        num_df = df.select_dtypes(include=[np.number])
        for col in num_df.columns:
            # Mean
            FEATURE_MEAN.labels(feature=col).set(float(np.nanmean(num_df[col].to_numpy())))

            # KS statistic
            baseline = get_baseline_for_feature(col)
            if baseline is not None:
                sample_sorted = np.sort(num_df[col].to_numpy(dtype=float))
                if sample_sorted.size > 0:
                    d = compute_ks_statistic(baseline, sample_sorted)
                    FEATURE_DRIFT_STAT.labels(feature=col).set(float(d))
    except Exception as e:
        logger.warning(f"Drift computation failed: {e}")

    # Online accuracy
    if y_true is not None:
        try:
            accuracy = update_accuracy(preds, y_true)
            MODEL_ACCURACY.set(accuracy)
        except Exception as e:
            logger.warning(f"Accuracy update failed: {e}")

    return PredictResponse(predictions=preds_list, model_version=str(model_version))


@router.post(
    "/survey/submit",
    response_model=SurveyResponse,
    summary="Submit customer survey",
)
async def submit_survey(input: InputCustomer, background_tasks: BackgroundTasks):
    """
    Soumission d'un formulaire client avec prédiction A/B testing

    - Prédit le churn avec A/B testing (80% Production, 20% Staging)
    - Enregistre dans data/production/production.csv
    - Push DVC en background
    """
    mlflow.set_experiment("Production_Customer_Churn_API")

    # Prédiction avec A/B testing
    df = pd.DataFrame([input.model_dump()])
    result = predict_single_ab(df)

    # Log MLflow
    mlflow.log_metric("latency", result["latency"])
    mlflow.set_tag("model_used", result["model"])

    # Prometheus metrics
    versions = get_model_versions()
    model_version = str(versions["production_model_version"] or "unknown")
    PREDICTION_LATENCY.labels(model_version=model_version).observe(result["latency"])
    PREDICTION_REQUESTS.labels(model_version=model_version).inc()

    # Sauvegarder dans production.csv
    file_path = Path(settings.DATA_PATH)
    customer_data = input.model_dump(by_alias=True)
    customer_data["Churn"] = int(result["prediction"])

    df_new = pd.DataFrame([customer_data])

    # Lire CSV existant
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

    if file_path.exists() and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=csv_columns)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_existing = pd.DataFrame(columns=csv_columns)

    # Combiner et sauvegarder
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(file_path, index=False)

    # DVC push en background
    background_tasks.add_task(_dvc_push_background)

    return SurveyResponse(success="Thank you for your submission")


async def _dvc_push_background():
    """Push DVC en arrière-plan"""
    import asyncio

    process = await asyncio.create_subprocess_exec(
        "dvc",
        "push",
        "-v",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        logger.info(f"[DVC] Push successful: {stdout.decode()}")
    else:
        logger.error(f"[DVC] Push failed: {stderr.decode()}")
