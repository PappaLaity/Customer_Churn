
# import os
# import time
# from pathlib import Path
# from datetime import datetime

# import mlflow
# import numpy as np
# import pandas as pd
# from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
# from prometheus_client import Counter, Gauge, Histogram
# from sqlalchemy.orm import Session 

# from src.api.core.config import settings
# from src.api.core.database import get_db  # <-- Utilisez get_db
# from src.api.core.logger import api_logger as logger
# from src.api.core.security import verify_api_key
# from src.api.entities.customerInput import InputCustomer
# from src.api.models.prediction import Prediction  # <-- Importez depuis core.models
# from src.api.schemas.predict import (
#     PredictPayload,
#     PredictResponse,
#     SurveyResponse,
# )
# from src.api.services.ml_service import (
#     get_model_versions,
#     predict_batch,
#     predict_single_ab,
# )
# from src.api.services.monitoring_service import (
#     compute_ks_statistic,
#     get_baseline_for_feature,
#     update_accuracy,
# )

# router = APIRouter(prefix="/predict", tags=["Predictions"])

# # Prometheus metrics (inchangé)
# PREDICTION_LATENCY = Histogram(
#     "prediction_latency_seconds", "Prediction latency", ["model_version"]
# )
# PREDICTION_REQUESTS = Counter(
#     "prediction_requests_total", "Total prediction requests", ["model_version"]
# )
# PREDICTION_ERRORS = Counter(
#     "prediction_errors_total", "Prediction errors", ["model_version", "error_type"]
# )
# FEATURE_DRIFT_STAT = Gauge(
#     "feature_drift_statistic",
#     "KS statistic for numeric features",
#     ["feature"],
# )
# FEATURE_MEAN = Gauge("feature_mean", "Online mean of numeric features", ["feature"])
# MODEL_ACCURACY = Gauge("model_accuracy", "Cumulative online accuracy")


# @router.post("", dependencies=[Depends(verify_api_key)], response_model=PredictResponse)
# async def predict_endpoint(
#     payload: PredictPayload,
#     db: Session = Depends(get_db)  # <-- Utilise get_db
# ):
#     """
#     Prédiction batch avec monitoring de drift et stockage DB
#     """
#     model_version = get_model_versions()["pyfunc_model_version"]

#     # Build DataFrame
#     try:
#         df = pd.DataFrame(payload.instances)
#     except Exception as e:
#         PREDICTION_ERRORS.labels(model_version=model_version, error_type="bad_input").inc()
#         raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

#     # Séparer les labels si fournis
#     y_true = None
#     if payload.label_key and payload.label_key in df.columns:
#         y_true = df[payload.label_key].to_numpy()
#         df = df.drop(columns=[payload.label_key])

#     # Prédiction
#     start = time.time()
#     try:
#         preds = predict_batch(df)
#         preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
#         duration = time.time() - start

#         PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)
#         PREDICTION_REQUESTS.labels(model_version=model_version).inc()

#     except Exception as e:
#         PREDICTION_ERRORS.labels(model_version=model_version, error_type="inference").inc()
#         raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

#     # ═══════════════════════════════════════════════════════════════
#     # Stocker dans PostgreSQL 
#     # ═══════════════════════════════════════════════════════════════
#     try:
#         for idx, row in df.iterrows():
#             prediction_record = Prediction( # Utilisez le modèle SQLModel importé
#                 customer_id=row.get('customer_id', None),
#                 tenure=int(row.get('tenure', 0)),
#                 monthly_charges=float(row.get('MonthlyCharges', row.get('monthly_charges', 0))),
#                 total_charges=float(row.get('TotalCharges', row.get('total_charges', 0))),
#                 internet_service_fiber_optic=bool(row.get('InternetService_Fiber_optic', False)),
#                 contract_two_year=bool(row.get('Contract_Two_year', False)),
#                 payment_method_electronic_check=bool(row.get('PaymentMethod_Electronic_check', False)),
#                 no_internet_service=bool(row.get('No_internet_service', False)),
#                 paperless_billing=bool(row.get('PaperlessBilling', False)),
#                 prediction=int(preds[idx]) if isinstance(preds, (list, np.ndarray)) else int(preds),
#                 model_version=str(model_version),
#             )
#             db.add(prediction_record)
        
#         db.commit()
#         logger.info(f"✅ Stored {len(df)} predictions in database")
    
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Failed to store predictions in DB: {e}")

#     # Drift computation (inchangé)
#     try:
#         num_df = df.select_dtypes(include=[np.number])
#         for col in num_df.columns:
#             FEATURE_MEAN.labels(feature=col).set(float(np.nanmean(num_df[col].to_numpy())))

#             baseline = get_baseline_for_feature(col)
#             if baseline is not None:
#                 sample_sorted = np.sort(num_df[col].to_numpy(dtype=float))
#                 if sample_sorted.size > 0:
#                     d = compute_ks_statistic(baseline, sample_sorted)
#                     FEATURE_DRIFT_STAT.labels(feature=col).set(float(d))
#     except Exception as e:
#         logger.warning(f"Drift computation failed: {e}")

#     # Online accuracy (inchangé)
#     if y_true is not None:
#         try:
#             accuracy = update_accuracy(preds, y_true)
#             MODEL_ACCURACY.set(accuracy)
#         except Exception as e:
#             logger.warning(f"Accuracy update failed: {e}")

#     return PredictResponse(predictions=preds_list, model_version=str(model_version))


# @router.post(
#     "/survey/submit",
#     response_model=SurveyResponse,
#     summary="Submit customer survey",
# )
# async def submit_survey(
#     input: InputCustomer, 
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)  # <-- Utilise get_db
# ):
#     """
#     Soumission d'un formulaire client avec prédiction A/B testing
#     """
#     mlflow.set_experiment("Production_Customer_Churn_API")

#     # Prédiction avec A/B testing (inchangé)
#     df = pd.DataFrame([input.model_dump()])
#     result = predict_single_ab(df)

#     # Log MLflow (inchangé)
#     mlflow.log_metric("latency", result["latency"])
#     mlflow.set_tag("model_used", result["model"])



#     # Prometheus metrics (inchangé)
#     versions = get_model_versions()
#     model_version = str(versions["production_model_version"] or "unknown")
#     PREDICTION_LATENCY.labels(model_version=model_version).observe(result["latency"])
#     PREDICTION_REQUESTS.labels(model_version=model_version).inc()

#     # ═══════════════════════════════════════════════════════════════
#     # Stocker dans PostgreSQL 
#     # ═══════════════════════════════════════════════════════════════
#     try:
#         customer_data = input.model_dump(by_alias=True)
        
#         prediction_record = Prediction( # Utilisez le modèle SQLModel importé
#             customer_id=customer_data.get('customerID', None),
#             tenure=int(customer_data.get('tenure', 0)),
#             monthly_charges=float(customer_data.get('MonthlyCharges', 0)),
#             total_charges=float(customer_data.get('TotalCharges', 0)),
#             internet_service_fiber_optic=bool(customer_data.get('InternetService_Fiber_optic', False)),
#             contract_two_year=bool(customer_data.get('Contract_Two_year', False)),
#             payment_method_electronic_check=bool(customer_data.get('PaymentMethod_Electronic_check', False)),
#             no_internet_service=bool(customer_data.get('No_internet_service', False)),
#             paperless_billing=bool(customer_data.get('PaperlessBilling', False)),
#             prediction=int(result["prediction"]),
#             probability=result["probability"],
#             model_version=result["model"],
#             model_stage=settings.MODEL_STAGE,
#             latency=result["latency"],
#         )
        
#         db.add(prediction_record)
#         db.commit()
#         db.refresh(prediction_record)
        
#         logger.info(
#             f"✅ Stored survey prediction in DB: ID={prediction_record.id}, "
#             f"prediction={result['prediction']}, model={result['model']}"
#         )
    
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Failed to store survey prediction: {e}")

#     # ═══════════════════════════════════════════════════════════════
#     # Sauvegarder dans CSV (GARDER pour compatibilité/backup)
#     # ═══════════════════════════════════════════════════════════════
#     file_path = Path(settings.DATA_PATH)
#     customer_data["Churn"] = int(result["prediction"])
#     df_new = pd.DataFrame([customer_data])

#     csv_columns = [
#         "tenure", "InternetService_Fiber_optic", "Contract_Two_year", 
#         "PaymentMethod_Electronic_check", "No_internet_service", "TotalCharges", 
#         "MonthlyCharges", "PaperlessBilling", "Churn",
#     ]

#     if file_path.exists() and os.path.getsize(file_path) > 0:
#         try:
#             df_existing = pd.read_csv(file_path)
#         except pd.errors.EmptyDataError:
#             df_existing = pd.DataFrame(columns=csv_columns)
#     else:
#         file_path.parent.mkdir(parents=True, exist_ok=True)
#         df_existing = pd.DataFrame(columns=csv_columns)

#     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#     df_combined.to_csv(file_path, index=False)

#     # DVC push en background (inchangé)
#     background_tasks.add_task(_dvc_push_background)

#     return SurveyResponse(success="Thank you for your submission")


# # ═══════════════════════════════════════════════════════════════
# # Endpoints pour consulter les prédictions
# # ═══════════════════════════════════════════════════════════════

# @router.get("/recent", dependencies=[Depends(verify_api_key)])
# async def get_recent_predictions(
#     limit: int = 100,
#     db: Session = Depends(get_db) # <-- Utilise get_db
# ):
#     """
#     Récupère les prédictions récentes depuis la base de données.
#     """
#     predictions = db.query(Prediction).order_by(
#         Prediction.created_at.desc()
#     ).limit(limit).all()
    
#     return {
#         "total": len(predictions),
#         # Assurez-vous que Prediction a une méthode to_dict() si nécessaire, 
#         # sinon utilisez pydantic depuis un schéma de réponse pour la conversion.
#         # "predictions": [p.to_dict() for p in predictions] 
#         "predictions": predictions
#     }


# @router.get("/stats", dependencies=[Depends(verify_api_key)])
# async def get_prediction_stats(
#     days: int = 7,
#     db: Session = Depends(get_db) # <-- Utilise get_db
# ):
#     """
#     Statistiques sur les prédictions récentes.
#     """
#     from datetime import timedelta
#     from sqlalchemy import func
    
#     cutoff_date = datetime.utcnow() - timedelta(days=days)
    
#     # Compter les prédictions
#     total = db.query(func.count(Prediction.id)).filter(
#         Prediction.created_at >= cutoff_date
#     ).scalar()
    
#     # Taux de churn prédit
#     churn_count = db.query(func.count(Prediction.id)).filter(
#         Prediction.created_at >= cutoff_date,
#         Prediction.prediction == 1
#     ).scalar()
    
#     churn_rate = (churn_count / total * 100) if total > 0 else 0
    
#     # Moyennes des features
#     avg_tenure = db.query(func.avg(Prediction.tenure)).filter(
#         Prediction.created_at >= cutoff_date
#     ).scalar()
    
#     avg_charges = db.query(func.avg(Prediction.monthly_charges)).filter(
#         Prediction.created_at >= cutoff_date
#     ).scalar()
    
#     return {
#         "period_days": days,
#         "total_predictions": total,
#         "churn_predictions": churn_count,
#         "churn_rate_percent": round(churn_rate, 2),
#         "avg_tenure_months": round(float(avg_tenure or 0), 1),
#         "avg_monthly_charges": round(float(avg_charges or 0), 2),
#     }


# # ═══════════════════════════════════════════════════════════════
# # Fonction helper (inchangée)
# # ═══════════════════════════════════════════════════════════════

# async def _dvc_push_background():
#     """Push DVC en arrière-plan"""
#     import asyncio

#     process = await asyncio.create_subprocess_exec(
#         "dvc",
#         "push",
#         "-v",
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#     )
#     stdout, stderr = await process.communicate()

#     if process.returncode == 0:
#         logger.info(f"[DVC] Push successful: {stdout.decode()}")
#     else:
#         logger.error(f"[DVC] Push failed: {stderr.decode()}")
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from sqlmodel import Session, select, func

from src.api.core.config import settings
from src.api.core.database import get_db
from src.api.core.logger import api_logger as logger
from src.api.core.security import verify_api_key
from src.api.entities.customerInput import InputCustomer
from src.api.models.prediction import Prediction
from src.api.schemas.predict import PredictPayload, PredictResponse, SurveyResponse
from src.api.services.ml_service import get_model_versions, predict_batch, predict_single_ab
from src.api.services.monitoring_service import compute_ks_statistic, get_baseline_for_feature, update_accuracy

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Prometheus metrics
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["model_version"])
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests", ["model_version"])
PREDICTION_ERRORS = Counter("prediction_errors_total", "Prediction errors", ["model_version", "error_type"])
FEATURE_DRIFT_STAT = Gauge("feature_drift_statistic", "KS statistic for numeric features", ["feature"])
FEATURE_MEAN = Gauge("feature_mean", "Online mean of numeric features", ["feature"])
MODEL_ACCURACY = Gauge("model_accuracy", "Cumulative online accuracy")


@router.post("", dependencies=[Depends(verify_api_key)], response_model=PredictResponse)
async def predict_endpoint(payload: PredictPayload, db: Session = Depends(get_db)):
    """
    Prédiction batch avec monitoring de drift et stockage SQLModel
    """
    model_version = get_model_versions()["pyfunc_model_version"]

    # Build DataFrame
    try:
        df = pd.DataFrame(payload.instances)
    except Exception as e:
        PREDICTION_ERRORS.labels(model_version=model_version, error_type="bad_input").inc()
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Séparer labels si fournis
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

    # Stockage SQLModel
    try:
        for idx, row in df.iterrows():
            record = Prediction(
                customer_id=row.get("customer_id"),
                tenure=int(row.get("tenure", 0)),
                monthly_charges=float(row.get("MonthlyCharges", row.get("monthly_charges", 0))),
                total_charges=float(row.get("TotalCharges", row.get("total_charges", 0))),
                internet_service_fiber_optic=bool(row.get("InternetService_Fiber_optic", False)),
                contract_two_year=bool(row.get("Contract_Two_year", False)),
                payment_method_electronic_check=bool(row.get("PaymentMethod_Electronic_check", False)),
                no_internet_service=bool(row.get("No_internet_service", False)),
                paperless_billing=bool(row.get("PaperlessBilling", False)),
                prediction=int(preds[idx]) if isinstance(preds, (list, np.ndarray)) else int(preds),
                model_version=str(model_version),
            )
            db.add(record)
        db.commit()
        logger.info(f"✅ Stored {len(df)} predictions in database")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to store predictions in DB: {e}")

    # Drift computation
    try:
        num_df = df.select_dtypes(include=[np.number])
        for col in num_df.columns:
            FEATURE_MEAN.labels(feature=col).set(float(np.nanmean(num_df[col].to_numpy())))
            baseline = get_baseline_for_feature(col)
            if baseline is not None and num_df[col].size > 0:
                d = compute_ks_statistic(np.sort(baseline), np.sort(num_df[col].to_numpy()))
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


@router.post("/survey/submit", response_model=SurveyResponse, summary="Submit customer survey")
async def submit_survey(input: InputCustomer, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Soumission formulaire client avec A/B testing
    """
    mlflow.set_experiment("Production_Customer_Churn_API")

    df = pd.DataFrame([input.model_dump()])
    result = predict_single_ab(df)

    mlflow.log_metric("latency", result["latency"])
    mlflow.set_tag("model_used", result["model"])

    # Metrics
    versions = get_model_versions()
    model_version = str(versions.get("production_model_version", "unknown"))
    PREDICTION_LATENCY.labels(model_version=model_version).observe(result["latency"])
    PREDICTION_REQUESTS.labels(model_version=model_version).inc()

    # Stockage SQLModel
    try:
        customer_data = input.model_dump(by_alias=True)
        record = Prediction(
            customer_id=customer_data.get("customerID"),
            tenure=int(customer_data.get("tenure", 0)),
            monthly_charges=float(customer_data.get("MonthlyCharges", 0)),
            total_charges=float(customer_data.get("TotalCharges", 0)),
            internet_service_fiber_optic=bool(customer_data.get("InternetService_Fiber_optic", False)),
            contract_two_year=bool(customer_data.get("Contract_Two_year", False)),
            payment_method_electronic_check=bool(customer_data.get("PaymentMethod_Electronic_check", False)),
            no_internet_service=bool(customer_data.get("No_internet_service", False)),
            paperless_billing=bool(customer_data.get("PaperlessBilling", False)),
            prediction=int(result["prediction"]),
            probability=result.get("probability"),
            model_version=result["model"],
            model_stage=settings.MODEL_STAGE,
            latency=result["latency"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"✅ Stored survey prediction in DB: ID={record.id}")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to store survey prediction: {e}")

    # Backup CSV
    file_path = Path(settings.DATA_PATH)
    customer_data["Churn"] = int(result["prediction"])
    df_new = pd.DataFrame([customer_data])
    csv_columns = [
        "tenure", "InternetService_Fiber_optic", "Contract_Two_year",
        "PaymentMethod_Electronic_check", "No_internet_service", "TotalCharges",
        "MonthlyCharges", "PaperlessBilling", "Churn",
    ]

    if file_path.exists() and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=csv_columns)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_existing = pd.DataFrame(columns=csv_columns)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(file_path, index=False)

    # DVC push en background
    background_tasks.add_task(_dvc_push_background)

    return SurveyResponse(success="Thank you for your submission")


@router.get("/recent", dependencies=[Depends(verify_api_key)])
async def get_recent_predictions(limit: int = 100, db: Session = Depends(get_db)):
    """
    Récupère les prédictions récentes
    """
    statement = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    predictions = db.exec(statement).all()
    return {"total": len(predictions), "predictions": predictions}


@router.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_prediction_stats(days: int = 7, db: Session = Depends(get_db)):
    """
    Statistiques sur les prédictions récentes
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    total = db.exec(select(func.count(Prediction.id)).where(Prediction.created_at >= cutoff)).one()
    churn_count = db.exec(select(func.count(Prediction.id)).where(Prediction.created_at >= cutoff, Prediction.prediction == 1)).one()
    avg_tenure = db.exec(select(func.avg(Prediction.tenure)).where(Prediction.created_at >= cutoff)).one()
    avg_charges = db.exec(select(func.avg(Prediction.monthly_charges)).where(Prediction.created_at >= cutoff)).one()

    churn_rate = (churn_count / total * 100) if total else 0

    return {
        "period_days": days,
        "total_predictions": total,
        "churn_predictions": churn_count,
        "churn_rate_percent": round(churn_rate, 2),
        "avg_tenure_months": round(float(avg_tenure or 0), 1),
        "avg_monthly_charges": round(float(avg_charges or 0), 2),
    }


async def _dvc_push_background():
    """Push DVC en arrière-plan"""
    import asyncio
    process = await asyncio.create_subprocess_exec(
        "dvc", "push", "-v",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        logger.info(f"[DVC] Push successful: {stdout.decode()}")
    else:
        logger.error(f"[DVC] Push failed: {stderr.decode()}")
