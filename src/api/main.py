"""FastAPI Customer Churn Prediction API.

Main application entry point with minimal orchestration logic.
All business logic has been extracted into specialized modules.
"""

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

from src.api.core.lifespan import lifespan
from src.api.routes import (
    ab_testing,
    auth,
    data,
    models,
    monitoring,
    predictions,
    users,
)

origins = [
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "https://customer-churn-dusky.vercel.app"
]



# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction",
    description="ML platform for customer churn prediction with A/B testing",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, #["*"], #origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)


# Root endpoint
@app.get("/", tags=["root"])
async def home():
    """API root endpoint."""
    return {"msg": "Customer Churn System"}


# Include all routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(ab_testing.router)
app.include_router(data.router)
