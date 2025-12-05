"""
Routes de monitoring (baseline, drift)
"""

from fastapi import APIRouter, Depends

from src.api.core.security import verify_api_key
from src.api.schemas.predict import BaselinePayload
from src.api.services.monitoring_service import (
    get_baseline_features,
    set_baseline,
)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.post("/baseline", dependencies=[Depends(verify_api_key)])
async def set_monitoring_baseline(payload: BaselinePayload):
    """
    Définit le baseline pour la détection de drift

    Body:
        numeric: Dict {feature_name: [list of values]}

    Example:
        {
            "numeric": {
                "tenure": [1, 5, 10, 15, 20],
                "MonthlyCharges": [50.0, 70.0, 90.0]
            }
        }
    """
    features = set_baseline(payload.numeric)
    return {"status": "ok", "features": features}


@router.get("/baseline", dependencies=[Depends(verify_api_key)])
async def get_monitoring_baseline():
    """Retourne la liste des features dans le baseline"""
    return {"features": get_baseline_features()}
