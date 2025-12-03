"""
Schemas Pydantic pour les prédictions et le monitoring
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictPayload(BaseModel):
    """Payload pour les prédictions batch"""

    instances: List[Dict[str, Any]]
    return_proba: Optional[bool] = False
    label_key: Optional[str] = None


class BaselinePayload(BaseModel):
    """Payload pour définir le baseline de monitoring"""

    numeric: Dict[str, List[float]] = {}


class PredictResponse(BaseModel):
    """Réponse de prédiction"""

    predictions: List[int]
    model_version: str


class SurveyResponse(BaseModel):
    """Réponse après soumission d'un formulaire"""

    success: str
