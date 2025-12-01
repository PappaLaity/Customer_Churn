"""Monitoring and health check endpoints.

This router handles:
- Drift detection baseline management
- Health checks
- Feature monitoring
"""

import numpy as np
from fastapi import APIRouter, Depends, Request

from src.api.core.security import verify_api_key
from src.api.entities.schemas import BaselinePayload


router = APIRouter(tags=["monitoring"])


@router.post("/monitoring/baseline", dependencies=[Depends(verify_api_key)])
async def set_baseline(payload: BaselinePayload, request: Request):
    """Set baseline data for drift detection.
    
    Args:
        payload: Baseline data with numeric features
        request: FastAPI request object
        
    Returns:
        Status and list of baseline features
    """
    app_state = request.app.state.app_state
    app_state.baseline_numeric_sorted = {
        f: np.sort(np.asarray(vals, dtype=float)) 
        for f, vals in payload.numeric.items()
    }
    return {
        "status": "ok", 
        "features": list(app_state.baseline_numeric_sorted.keys())
    }


@router.get("/monitoring/baseline", dependencies=[Depends(verify_api_key)])
async def get_baseline(request: Request):
    """Get current drift detection baseline features.
    
    Args:
        request: FastAPI request object
        
    Returns:
        List of baseline feature names
    """
    app_state = request.app.state.app_state
    return {"features": list(app_state.baseline_numeric_sorted.keys())}


@router.get("/health", dependencies=[Depends(verify_api_key)])
async def check_health():
    """Health check endpoint.
    
    Returns:
        Simple health status message
    """
    return {"check": "I'm ok! No worry"}
