"""Model management endpoints.

This router handles:
- Model version information
- Model registry queries
"""

import os

import mlflow
from fastapi import APIRouter, Depends, Request

from src.api.core.security import verify_api_key


router = APIRouter(tags=["models"])

# Environment variables
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")


@router.get("/model/version", dependencies=[Depends(verify_api_key)])
async def get_model_version(request: Request):
    """Get currently loaded model versions.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Production and staging model versions
    """
    app_state = request.app.state.app_state
    return {
        "production_model_version": app_state.prod_version,
        "staging_model_version": app_state.stag_version,
    }


@router.get("/models", dependencies=[Depends(verify_api_key)])
async def get_models():
    """Get all registered model versions from MLflow.
    
    Returns:
        List of all model versions with metadata
    """
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
                "description": m.description,
                "model_name": m.tags.get("model_name"),
                "cv_mean": m.tags.get("cv_mean"),
                "test_accuracy": m.tags.get("test_accuracy"),
            }
            for m in models
        ]
    }
