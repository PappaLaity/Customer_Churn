"""API request/response schemas.

This module contains all Pydantic models used for API requests and responses,
extracted from main.py for better organization.
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class PredictPayload(BaseModel):
    """Payload for batch prediction endpoint.
    
    Attributes:
        instances: List of feature dictionaries to predict on
        return_proba: Whether to return prediction probabilities
        label_key: Optional column name containing true labels for accuracy tracking
    """
    instances: List[Dict[str, Any]]
    return_proba: Optional[bool] = False
    label_key: Optional[str] = None


class BaselinePayload(BaseModel):
    """Payload for setting drift detection baseline.
    
    Attributes:
        numeric: Dictionary mapping feature names to lists of baseline values
    """
    numeric: Dict[str, List[float]] = {}


class AbConfigUpdate(BaseModel):
    """Payload for updating A/B test configuration.
    
    Attributes:
        enabled: Whether A/B testing is enabled
        bucket_b_pct: Percentage of traffic to send to bucket B (0.0 to 1.0)
        sticky_header: HTTP header name to use for sticky bucket assignment
    """
    enabled: Optional[bool] = None
    bucket_b_pct: Optional[float] = None
    sticky_header: Optional[str] = None
