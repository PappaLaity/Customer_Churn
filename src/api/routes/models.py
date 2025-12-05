"""
Routes pour la gestion des modèles MLflow
"""

import mlflow
from fastapi import APIRouter, Depends

from src.api.core.config import settings
from src.api.core.security import verify_api_key
from src.api.services.ml_service import get_model_versions

router = APIRouter(prefix="/model", tags=["Models"])


@router.get("/version", dependencies=[Depends(verify_api_key)])
async def get_current_version():
    """Retourne les versions des modèles actuellement chargés"""
    return get_model_versions()


@router.get("s")  # /models
async def list_all_models():
    """Liste tous les modèles disponibles dans MLflow Registry"""
    try:
        if settings.ENV == "test":
            return {"models": []}

        models = mlflow.search_model_versions(
            filter_string=f"name='{settings.MODEL_NAME}'", max_results=1000
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
    except Exception:
        return {"models": []}
