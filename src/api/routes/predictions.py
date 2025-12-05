"""Prediction endpoints.

This router handles:
- Batch predictions via /predict  
- CSV file predictions via /predict/csv
- Single predictions with data logging via /survey/submit
- A/B testing prediction logic
"""

import io
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from mlflow.tracking import MlflowClient
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from src.api.core.database import get_session
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
from src.api.entities.predictions import BatchPrediction, BatchPredictionRead, BatchSummary
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
        
        # Get probabilities if model supports it
        probabilities = None
        if hasattr(app_state.pyfunc_model, '_model_impl'):
            underlying_model = app_state.pyfunc_model._model_impl
            if hasattr(underlying_model, 'predict_proba'):
                try:
                    proba = underlying_model.predict_proba(df)
                    # Get probability of class 1 (churn)
                    probabilities = [float(p[1]) for p in proba]
                except Exception:
                    pass
        
        # Try direct predict_proba on pyfunc model
        if probabilities is None and hasattr(app_state.pyfunc_model, 'predict_proba'):
            try:
                proba = app_state.pyfunc_model.predict_proba(df)
                probabilities = [float(p[1]) for p in proba]
            except Exception:
                pass
        
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

    response = {
        "predictions": preds_list,
        "model_version": model_version
    }
    
    # Add probabilities if available
    if probabilities is not None:
        response["probabilities"] = probabilities
    
    return response


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
    churn_prediction = result["prediction"]
    churn_probability = result.get("probability", 0.5)
    
    return {
        "success": "Thank you for your submission",
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
        "will_churn": bool(churn_prediction),
        "message": "Customer likely to churn" if churn_prediction == 1 else "Customer likely to stay",
        "model_version": result.get("model_version")
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
    probability = 0.5  # Default probability
    
    if bucket == "A" and app_state.model_A is not None:
        preds = app_state.model_A.predict(df)
        # Get probability if model supports it
        if hasattr(app_state.model_A, 'predict_proba'):
            proba = app_state.model_A.predict_proba(df)
            probability = float(proba[0][1])  # Probability of class 1 (churn)
        model_used = f"Production(v{app_state.prod_version})"
        model_version = str(app_state.prod_version)
        
    elif bucket == "B" and app_state.model_B is not None:
        preds = app_state.model_B.predict(df)
        # Get probability if model supports it
        if hasattr(app_state.model_B, 'predict_proba'):
            proba = app_state.model_B.predict_proba(df)
            probability = float(proba[0][1])  # Probability of class 1 (churn)
        model_used = f"Staging(v{app_state.stag_version})"
        model_version = str(app_state.stag_version)
        
    else:
        # Fallback to PyFunc model
        _ensure_model_loaded(request)
        preds = app_state.pyfunc_model.predict(df)
        # PyFunc models may not have predict_proba, use prediction as probability
        probability = float(preds[0]) if preds[0] in [0, 1] else 0.5
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
        "probability": probability,
        "model_version": model_version,
        "latency": latency
    }


# ============================================================================
# CSV Batch Prediction Endpoints
# ============================================================================

REQUIRED_COLUMNS = [
    "tenure", "InternetService_Fiber_optic", "Contract_Two_year",
    "PaymentMethod_Electronic_check", "No_internet_service",
    "TotalCharges", "MonthlyCharges", "PaperlessBilling"
]


@router.post("/predict/csv", dependencies=[Depends(verify_api_key)])
async def predict_csv(
    file: UploadFile = File(...),
    request: Request = None,
    session: Session = Depends(get_session)
):
    """Upload CSV file and get batch predictions.
    
    Protected endpoint requiring API key authentication.
    
    The CSV must have the following columns:
    - tenure (float, 0-120)
    - InternetService_Fiber_optic (bool/int)
    - Contract_Two_year (bool/int)
    - PaymentMethod_Electronic_check (bool/int)
    - No_internet_service (int, 0/1)
    - TotalCharges (float)
    - MonthlyCharges (float)
    - PaperlessBilling (int, 0/1)
    
    Args:
        file: CSV file upload
        request: FastAPI request
        session: Database session
        
    Returns:
        Batch predictions with batch_id for retrieval
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read CSV content
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")
    
    # Validate columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns: {list(missing_cols)}"
        )
    
    # Keep only required columns
    df = df[REQUIRED_COLUMNS]
    
    # Validate data types and ranges
    try:
        df["tenure"] = df["tenure"].astype(float)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["No_internet_service"] = df["No_internet_service"].astype(int)
        df["PaperlessBilling"] = df["PaperlessBilling"].astype(int)
        
        # Convert string booleans to actual booleans
        for col in ["InternetService_Fiber_optic", "Contract_Two_year", "PaymentMethod_Electronic_check"]:
            if df[col].dtype == object:
                df[col] = df[col].map({'true': True, 'false': False, 'True': True, 'False': False, '1': True, '0': False})
            df[col] = df[col].astype(bool)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data validation error: {e}")
    
    # Check for NaN values
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise HTTPException(
            status_code=400,
            detail=f"CSV contains null values in columns: {null_cols}"
        )
    
    # Ensure model is loaded
    _ensure_model_loaded(request)
    app_state = request.app.state.app_state
    
    # Generate batch ID
    batch_id = str(uuid.uuid4())
    
    # Run predictions
    start = time.time()
    try:
        preds = app_state.pyfunc_model.predict(df)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        
        # Get probabilities
        probabilities = [0.5] * len(preds_list)  # Default
        if hasattr(app_state.pyfunc_model, '_model_impl'):
            underlying_model = app_state.pyfunc_model._model_impl
            if hasattr(underlying_model, 'predict_proba'):
                try:
                    proba = underlying_model.predict_proba(df)
                    probabilities = [float(p[1]) for p in proba]
                except Exception:
                    pass
        
        model_version = str(app_state.pyfunc_model_version)
        duration = time.time() - start
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Store predictions in database
    predictions_response = []
    for idx, (pred, prob) in enumerate(zip(preds_list, probabilities)):
        input_row = df.iloc[idx].to_dict()
        
        # Convert numpy types to Python types for JSON
        input_row = {k: (bool(v) if isinstance(v, (np.bool_, )) else 
                        float(v) if isinstance(v, (np.floating,)) else
                        int(v) if isinstance(v, (np.integer,)) else v)
                    for k, v in input_row.items()}
        
        batch_pred = BatchPrediction(
            batch_id=batch_id,
            row_index=idx,
            input_data=input_row,
            prediction=int(pred),
            probability=prob,
            model_version=model_version
        )
        session.add(batch_pred)
        
        predictions_response.append({
            "row_index": idx,
            "prediction": int(pred),
            "probability": round(prob, 4),
            "will_churn": bool(pred)
        })
    
    session.commit()
    
    # Calculate summary stats
    churn_count = sum(1 for p in preds_list if p == 1)
    avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0
    
    return {
        "batch_id": batch_id,
        "total_rows": len(preds_list),
        "churn_count": churn_count,
        "stay_count": len(preds_list) - churn_count,
        "avg_probability": round(avg_probability, 4),
        "model_version": model_version,
        "latency_seconds": round(duration, 3),
        "predictions": predictions_response
    }


@router.get("/predict/batch/{batch_id}", dependencies=[Depends(verify_api_key)])
async def get_batch_predictions(
    batch_id: str,
    session: Session = Depends(get_session)
):
    """Retrieve predictions for a specific batch.
    
    Args:
        batch_id: UUID of the batch to retrieve
        session: Database session
        
    Returns:
        List of predictions for the batch
    """
    statement = select(BatchPrediction).where(
        BatchPrediction.batch_id == batch_id
    ).order_by(BatchPrediction.row_index)
    
    results = session.exec(statement).all()
    
    if not results:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    
    predictions = []
    for r in results:
        predictions.append({
            "row_index": r.row_index,
            "input_data": r.input_data,
            "prediction": r.prediction,
            "probability": r.probability,
            "will_churn": bool(r.prediction),
            "created_at": r.created_at.isoformat()
        })
    
    # Calculate summary
    churn_count = sum(1 for p in predictions if p["prediction"] == 1)
    avg_prob = sum(p["probability"] for p in predictions) / len(predictions)
    
    return {
        "batch_id": batch_id,
        "total_rows": len(predictions),
        "churn_count": churn_count,
        "stay_count": len(predictions) - churn_count,
        "avg_probability": round(avg_prob, 4),
        "model_version": results[0].model_version,
        "created_at": results[0].created_at.isoformat(),
        "predictions": predictions
    }


@router.get("/predict/batches", dependencies=[Depends(verify_api_key)])
async def list_batches(
    limit: int = 20,
    session: Session = Depends(get_session)
):
    """List recent batch prediction jobs.
    
    Args:
        limit: Maximum number of batches to return (default 20)
        session: Database session
        
    Returns:
        List of batch summaries
    """
    # Get distinct batch_ids with their stats
    statement = select(BatchPrediction).order_by(
        BatchPrediction.created_at.desc()
    )
    results = session.exec(statement).all()
    
    # Group by batch_id
    batches = {}
    for r in results:
        if r.batch_id not in batches:
            batches[r.batch_id] = {
                "batch_id": r.batch_id,
                "total_rows": 0,
                "churn_count": 0,
                "model_version": r.model_version,
                "created_at": r.created_at.isoformat(),
                "probabilities": []
            }
        batches[r.batch_id]["total_rows"] += 1
        if r.prediction == 1:
            batches[r.batch_id]["churn_count"] += 1
        batches[r.batch_id]["probabilities"].append(r.probability)
    
    # Calculate averages and format response
    batch_list = []
    for b in list(batches.values())[:limit]:
        b["stay_count"] = b["total_rows"] - b["churn_count"]
        b["avg_probability"] = round(sum(b["probabilities"]) / len(b["probabilities"]), 4)
        del b["probabilities"]
        batch_list.append(b)
    
    return {"batches": batch_list, "count": len(batch_list)}
