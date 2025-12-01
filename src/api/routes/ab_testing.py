"""A/B testing configuration endpoints.

This router handles A/B test experiment configuration management.
"""

import os
from pathlib import Path

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


@router.get("/results", dependencies=[Depends(verify_api_key)])
async def get_ab_results(metric: str = "latency"):
    """Get A/B test statistical analysis results.
    
    Analyzes the exposure logs and provides statistical comparison
    between variant A (Production) and variant B (Staging).
    
    Args:
        metric: Metric to analyze ('latency' supported currently)
        
    Returns:
        Statistical analysis results including:
        - Sample sizes per variant
        - Metric values per variant
        - Lift percentage
        - Statistical significance (p-value)
        - Recommendation (PROMOTE/CONTINUE/ROLLBACK)
        
    Raises:
        HTTPException: If analysis fails or insufficient data
    """
    from src.experiments.ab_analysis import generate_report
    
    exposures_path = os.getenv("AB_EXPOSURES_PATH", "data/experiments/ab_exposures.csv")
    
    if not Path(exposures_path).exists():
        raise HTTPException(
            status_code=404,
            detail="No A/B test data found. Run some experiments first."
        )
    
    try:
        report = generate_report(
            exposures_path=exposures_path,
            metric=metric,
        )
        
        if report['status'] == 'error':
            raise HTTPException(
                status_code=400,
                detail=report['error']
            )
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
