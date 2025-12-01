"""A/B testing configuration endpoints.

This router handles A/B test experiment configuration management.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.core.security import verify_api_key
from src.api.entities.schemas import AbConfigUpdate


router = APIRouter(prefix="/ab", tags=["ab_testing"])


@router.get("/config", dependencies=[Depends(verify_api_key)])
async def get_ab_config(request: Request):
    """Get current A/B test configuration.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current A/B test configuration
    """
    cfg = request.app.state.app_state.ab_config
    return {
        "enabled": bool(cfg.enabled),
        "bucket_b_pct": float(cfg.bucket_b_pct),
        "sticky_header": cfg.sticky_header,
    }


@router.post("/config", dependencies=[Depends(verify_api_key)])
async def set_ab_config(update: AbConfigUpdate, request: Request):
    """Update A/B test configuration.
    
    Args:
        update: Configuration updates to apply
        request: FastAPI request object
        
    Returns:
        Updated configuration
        
    Raises:
        HTTPException: If bucket_b_pct is invalid
    """
    cfg = request.app.state.app_state.ab_config
    
    if update.enabled is not None:
        cfg.enabled = bool(update.enabled)
    
    if update.bucket_b_pct is not None:
        try:
            cfg.bucket_b_pct = float(update.bucket_b_pct)
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="bucket_b_pct must be a float"
            )
    
    if update.sticky_header is not None:
        cfg.sticky_header = str(update.sticky_header)
    
    # Clamp values to valid ranges
    cfg.clamp()
    
    return {
        "status": "ok",
        "config": {
            "enabled": bool(cfg.enabled),
            "bucket_b_pct": float(cfg.bucket_b_pct),
            "sticky_header": cfg.sticky_header,
        },
    }
