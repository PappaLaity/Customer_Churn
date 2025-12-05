"""Batch Prediction entities for database storage.

Stores predictions made from CSV uploads for auditing and retrieval.
"""

from datetime import datetime
from typing import Optional
from sqlmodel import Column, Field, SQLModel, String, DateTime, JSON
from sqlalchemy import func


class BatchPrediction(SQLModel, table=True):
    """Store individual predictions from batch CSV uploads."""
    
    __tablename__ = "batch_prediction"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: str = Field(
        sa_column=Column(String(36), nullable=False, index=True),
        description="UUID grouping predictions from same CSV upload"
    )
    row_index: int = Field(
        description="Row number in original CSV (0-indexed)"
    )
    input_data: dict = Field(
        sa_column=Column(JSON, nullable=False),
        description="Original input features as JSON"
    )
    prediction: int = Field(
        ge=0, le=1,
        description="Churn prediction (0=stay, 1=churn)"
    )
    probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability of churn (0.0-1.0)"
    )
    model_version: str = Field(
        sa_column=Column(String(50), nullable=True),
        description="Model version used for prediction"
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime, server_default=func.now(), nullable=False),
        description="Timestamp of prediction"
    )


class BatchPredictionRead(SQLModel):
    """Response model for batch prediction results."""
    id: int
    batch_id: str
    row_index: int
    input_data: dict
    prediction: int
    probability: float
    model_version: Optional[str]
    created_at: datetime


class BatchSummary(SQLModel):
    """Summary of a batch prediction job."""
    batch_id: str
    total_rows: int
    churn_count: int
    stay_count: int
    avg_probability: float
    model_version: Optional[str]
    created_at: datetime
