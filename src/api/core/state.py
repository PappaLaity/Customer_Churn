"""Centralized application state management.

This module defines the AppState class that holds all global state
for the FastAPI application, replacing module-level global variables.
"""

from typing import Any, Dict, Optional
import numpy as np


class AppState:
    """Centralized application state for the Customer Churn API.
    
    This class holds all stateful data that was previously stored in
    module-level global variables in main.py.
    """
    
    def __init__(self):
        # --- Model State ---
        # PyFunc model for batch predictions (fallback)
        self.pyfunc_model: Optional[Any] = None
        self.pyfunc_model_version: str = ""
        
        # A/B testing models (sklearn)
        self.model_A: Optional[Any] = None  # Production model
        self.model_B: Optional[Any] = None  # Staging model
        
        # Model version tracking
        self.prod_version: Optional[str] = None
        self.stag_version: Optional[str] = None
        self.prod_source: Optional[str] = None
        self.stag_source: Optional[str] = None
        
        # A/B test configuration
        self.ab_config: Optional[Any] = None  # ExperimentConfig instance
        
        # --- Metrics State ---
        # Online accuracy tracking
        self.total_with_label: int = 0
        self.correct_with_label: int = 0
        
        # Baseline for drift detection (feature -> sorted values)
        self.baseline_numeric_sorted: Dict[str, np.ndarray] = {}
