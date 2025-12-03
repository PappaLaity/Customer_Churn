"""Prediction endpoints.

This router handles:
- Batch predictions via /predict  
- Single predictions with data logging via /survey/submit
- A/B testing prediction logic
"""

import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from mlflow.tracking import MlflowClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.core.metrics import (
    FEATURE_DRIFT_STAT,
    FEATURE_MEAN,
    MODEL_ACCURACY,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
    compute_ks_statistic,
)
from src.api.core.security import verify_api_key
from src.api.entities.customerInput import InputCustomer
from src.api.entities.schemas import PredictPayload
from src.api.routes.data import dvc_push_background
from src.experiments.ab import assign_bucket, log_exposure


router = APIRouter(tags=["predictions"])

# Initialize rate limiter (used for public endpoints)
limiter = Limiter(key_func=get_remote_address)

# Environment variables
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _ensure_model_loaded(request: Request):
    """Ensure the PyFunc model is loaded.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If model cannot be loaded
    """
    app_state = request.app.state.app_state
    
    if app_state.pyfunc_model is not None:
        return
    
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        app_state.pyfunc_model = mlflow.pyfunc.load_model(uri)
        
        # Try to get version number
        try:
            client = MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if versions:
                app_state.pyfunc_model_version = versions[0].version
        except Exception:
            pass
            
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not available: {e}"
        )


@router.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(payload: PredictPayload, request: Request):
    """Batch prediction endpoint.
    
    Args:
        payload: Prediction payload with instances
        request: FastAPI request object
        
    Returns:
        Predictions and model version
        
    Raises:
        HTTPException: If input is invalid or inference fails
    """
    _ensure_model_loaded(request)
    app_state = request.app.state.app_state
    model_version = str(app_state.pyfunc_model_version)

    # Build dataframe
    try:
        df = pd.DataFrame(payload.instances)
    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_version=model_version, error_type="bad_input"
        ).inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Optionally separate labels for accuracy tracking
    y_true = None
    if payload.label_key and payload.label_key in df.columns:
        y_true = df[payload.label_key].to_numpy()
        df = df.drop(columns=[payload.label_key])

    # Predict
    start = time.time()
    try:
        preds = app_state.pyfunc_model.predict(df)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        duration = time.time() - start
        
        PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)
        PREDICTION_REQUESTS.labels(model_version=model_version).inc()
        
    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_version=model_version, error_type="inference"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Drift computation (numeric features only, if baseline exists)
    if app_state.baseline_numeric_sorted:
        try:
            num_df = df.select_dtypes(include=[np.number])
            for col in num_df.columns:
                # Update mean
                FEATURE_MEAN.labels(feature=col).set(
                    float(np.nanmean(num_df[col].to_numpy()))
                )
                
                # Compute drift if baseline exists for this feature
                if col in app_state.baseline_numeric_sorted:
                    sample_sorted = np.sort(num_df[col].to_numpy(dtype=float))
                    if sample_sorted.size > 0:
                        d = compute_ks_statistic(
                            app_state.baseline_numeric_sorted[col], 
                            sample_sorted
                        )
                        FEATURE_DRIFT_STAT.labels(feature=col).set(float(d))
        except Exception:
            pass  # Don't fail predictions if drift computation fails

    # Online accuracy tracking (if labels provided)
    if y_true is not None:
        try:
            correct = np.sum((np.asarray(preds) == np.asarray(y_true)).astype(int))
            app_state.total_with_label += len(preds_list)
            app_state.correct_with_label += int(correct)
            
            accuracy = app_state.correct_with_label / max(1, app_state.total_with_label)
            MODEL_ACCURACY.set(accuracy)
        except Exception:
            pass  # Don't fail predictions if accuracy tracking fails

    return {"predictions": preds_list, "model_version": model_version}


@router.post("/survey/submit")
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def submit_survey(
    input: InputCustomer, 
    background_tasks: BackgroundTasks, 
    request: Request
):
    """Submit customer survey and get churn prediction.
    
    Public endpoint with rate limiting (10 requests/minute per IP).
    
    This endpoint:
    1. Performs A/B tested prediction
    2. Logs the result to production CSV
    3. Triggers DVC push in background
    
    Args:
        input: Customer survey data
        background_tasks: FastAPI background tasks
        request: FastAPI request object
        
    Returns:
        Success confirmation
    """
    app_state = request.app.state.app_state
    
    # Predict using A/B testing
    start = time.time()
    result = await predict_single(input, request)
    duration = time.time() - start

    # Log metrics
    model_version_used = str(app_state.prod_version or app_state.pyfunc_model_version)
    PREDICTION_LATENCY.labels(model_version=model_version_used).observe(duration)
    PREDICTION_REQUESTS.labels(model_version=model_version_used).inc()

    # Append to production CSV
    file_path = Path("data/production/production.csv")
    df = pd.DataFrame([input.model_dump()])
    df["Churn"] = result["prediction"]
    
    if file_path.exists():
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(file_path, index=False)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

    # DVC push in background
    background_tasks.add_task(dvc_push_background)
    
    # Return prediction with interpretation
    # NOTE: This is a public endpoint - do not expose:
    # - Model version numbers
    # - Internal system paths
    # - Latency/timing (timing attacks)
    # - A/B test bucket assignment
    churn_prediction = result["prediction"]
    return {
        "success": "Thank you for your submission",
        "prediction": churn_prediction,
        "will_churn": bool(churn_prediction),
        "message": "Customer likely to churn" if churn_prediction == 1 else "Customer likely to stay"
        # REMOVED for security:
        # - "model_used": reveals internal model versions
        # - "latency": enables timing attacks
    }


async def predict_single(data: InputCustomer, request: Request) -> Dict[str, Any]:
    """Perform single prediction with A/B testing.
    
    Intelligently routes requests to Production (A) or Staging (B) models
    based on A/B testing configuration.
    
    Args:
        data: Customer input data
        request: FastAPI request object
        
    Returns:
        Prediction result with model info and latency
    """
    app_state = request.app.state.app_state
    df = pd.DataFrame([data.model_dump()])

    # Determine bucket assignment
    bucket, subject_id = assign_bucket(request, app_state.ab_config)
    
    # Fallback to randomized split if no subject_id
    if (subject_id is None and 
        app_state.model_A is not None and 
        app_state.model_B is not None):
        p_b = float(app_state.ab_config.bucket_b_pct)
        bucket = "B" if random.random() < p_b else "A"

    # Predict based on bucket
    start = time.time()
    
    if bucket == "A" and app_state.model_A is not None:
        preds = app_state.model_A.predict(df)
        model_used = f"Production(v{app_state.prod_version})"
        model_version = str(app_state.prod_version)
        
    elif bucket == "B" and app_state.model_B is not None:
        preds = app_state.model_B.predict(df)
        model_used = f"Staging(v{app_state.stag_version})"
        model_version = str(app_state.stag_version)
        
    else:
        # Fallback to PyFunc model
        _ensure_model_loaded(request)
        preds = app_state.pyfunc_model.predict(df)
        model_used = f"Registry({MODEL_STAGE})"
        model_version = str(app_state.pyfunc_model_version)
        
    latency = time.time() - start

    # Log exposure for A/B testing analytics
    try:
        log_exposure(
            endpoint="/survey/submit",
            subject_id=subject_id,
            bucket=bucket,
            model=model_used,
            model_version=model_version,
            latency_sec=latency,
        )
    except Exception:
        pass  # Don't fail prediction if exposure logging fails

    return {
        "model": model_used, 
        "prediction": int(preds[0]), 
        "latency": latency
    }
