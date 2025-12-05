"""
Service ML : Gestion des modèles MLflow
- Chargement des modèles Production/Staging
- A/B testing
- Prédictions
"""

import random
import time
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.core.config import settings
from src.api.core.logger import api_logger as logger

# Global model cache
_model_cache = {
    "pyfunc_model": None,
    "model_version": settings.MODEL_STAGE,
    "model_A": None,  # Production
    "model_B": None,  # Staging
    "prod_version": None,
    "prod_source": None,
    "stag_version": None,
    "stag_source": None,
}


class DummyModel:
    """Modèle factice pour les tests"""

    def predict(self, df):
        try:
            n = len(df)
        except Exception:
            n = 1
        return np.zeros(n, dtype=int)


def ensure_model_loaded() -> None:
    """
    S'assure que le modèle pyfunc est chargé
    Utilisé pour les prédictions batch
    """
    if _model_cache["pyfunc_model"] is not None:
        return

    uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
    try:
        if settings.ENV == "test":
            _model_cache["pyfunc_model"] = DummyModel()
            return

        _model_cache["pyfunc_model"] = mlflow.pyfunc.load_model(uri)

        # Récupérer la version
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            versions = client.get_latest_versions(
                settings.MODEL_NAME, stages=[settings.MODEL_STAGE]
            )
            if versions:
                _model_cache["model_version"] = versions[0].version
        except Exception:
            pass

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


def load_ab_models(model_name: str = None) -> None:
    """
    Charge les modèles Production et Staging pour l'A/B testing
    Appelé au démarrage et périodiquement
    """
    model_name = model_name or settings.MODEL_NAME

    try:
        if settings.ENV == "test":
            _model_cache.update(
                {
                    "prod_version": None,
                    "prod_source": None,
                    "stag_version": None,
                    "stag_source": None,
                    "model_A": None,
                    "model_B": None,
                }
            )
            return

        models = mlflow.search_model_versions(
            filter_string=f"name='{model_name}'", max_results=1000
        )

        # Trouver Production et Staging
        for m in models:
            if m.current_stage == "Production":
                _model_cache["prod_version"] = m.version
                _model_cache["prod_source"] = m.source
            if m.current_stage == "Staging":
                _model_cache["stag_version"] = m.version
                _model_cache["stag_source"] = m.source

    except Exception:
        _model_cache.update(
            {
                "prod_version": None,
                "prod_source": None,
                "stag_version": None,
                "stag_source": None,
            }
        )

    logger.info(
        f"Production model: v{_model_cache['prod_version']}, "
        f"source: {_model_cache['prod_source']}"
    )
    logger.info(
        f"Staging model: v{_model_cache['stag_version']}, " f"source: {_model_cache['stag_source']}"
    )

    # Charger les modèles sklearn
    try:
        if _model_cache["prod_source"]:
            _model_cache["model_A"] = mlflow.sklearn.load_model(_model_cache["prod_source"])
            logger.info(f"✅ Loaded Production model: {_model_cache['prod_source']}")

        if _model_cache["stag_source"]:
            _model_cache["model_B"] = mlflow.sklearn.load_model(_model_cache["stag_source"])
            logger.info(f"✅ Loaded Staging model: {_model_cache['stag_source']}")

    except Exception as e:
        logger.error(f"Error loading sklearn models: {e}")
        raise


def predict_batch(df: pd.DataFrame) -> np.ndarray:
    """
    Prédiction batch avec le modèle pyfunc
    Utilisé par /predict
    """
    ensure_model_loaded()
    return _model_cache["pyfunc_model"].predict(df)


def predict_single_ab(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prédiction single record avec A/B testing
    Utilisé par /survey/submit

    Returns:
        {
            "model": "Production(v2)" | "Staging(v3)" | "Registry(Production)",
            "prediction": 0 or 1,
            "latency": 0.05
        }
    """
    model_choice = "A"  # Default Production

    # Vérifier si A/B testing possible
    if _model_cache["model_A"] is not None and _model_cache["model_B"] is not None:
        model_choice = "A" if random.random() < settings.AB_TESTING_RATIO else "B"

    start = time.time()

    # Production model
    if model_choice == "A" and _model_cache["model_A"] is not None:
        preds = _model_cache["model_A"].predict(df)
        model_used = f"Production(v{_model_cache['prod_version']})"

    # Staging model
    elif model_choice == "B" and _model_cache["model_B"] is not None:
        preds = _model_cache["model_B"].predict(df)
        model_used = f"Staging(v{_model_cache['stag_version']})"

    # Fallback: pyfunc model
    else:
        ensure_model_loaded()
        preds = _model_cache["pyfunc_model"].predict(df)
        model_used = f"Registry({settings.MODEL_STAGE})"

    latency = time.time() - start

    return {
        "model": model_used,
        "prediction": int(preds[0]),
        "latency": latency,
    }


def get_model_versions() -> Dict[str, Optional[str]]:
    """Retourne les versions des modèles actuellement chargés"""
    return {
        "production_model_version": _model_cache["prod_version"],
        "staging_model_version": _model_cache["stag_version"],
        "pyfunc_model_version": _model_cache["model_version"],
    }
