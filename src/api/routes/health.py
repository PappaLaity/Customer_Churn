"""
Routes de health check
"""

from fastapi import APIRouter, Depends

from src.api.core.security import verify_api_key

router = APIRouter(tags=["Health"])


@router.get("/", summary="Home endpoint")
async def home():
    """Point d'entrée de l'API"""
    return {"msg": "Customer Churn System"}


@router.get("/health", dependencies=[Depends(verify_api_key)], summary="Health check")
async def check_health():
    """Vérification de l'état de l'API"""
    return {"check": "I'm ok! No worry"}
