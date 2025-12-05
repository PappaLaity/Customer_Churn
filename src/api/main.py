"""FastAPI Customer Churn Prediction API.

Main application entry point with minimal orchestration logic.
All business logic has been extracted into specialized modules.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
from src.monitoring.evidently_exporter import router as evidently_router

# CORS Configuration
# For production, add your domain to this list
origins = [
    "http://localhost:8081",           # Local development
    "http://127.0.0.1:8081",           # Local development (IP)
    "https://customer-churn-dusky.vercel.app"  # Production frontend
    # Add more production domains as needed
]


# Initialize rate limiter
# Uses IP address to track request rates
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ML platform for customer churn prediction with A/B testing.
    
    ## Security Model
    
    **Public Endpoints** (No auth required):
    - `/survey/submit` - Submit survey for prediction
    - `/health` - Health check
    - `/models` - View available models
    - `/auth/login` - Admin login
    
    **Protected Endpoints** (API key required):
    - `/predict` - Batch predictions
    - `/model/version` - Model versions
    - `/ab/*` - A/B testing management
    - `/monitoring/*` - Monitoring & drift detection
    - `/customers/infos` - Customer data
    - `/users/*` - User management
    
    See `/health` for API status and `docs/API_SECURITY.md` for full documentation.
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# Attach rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Global exception handler to prevent information leakage
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions with generic error message."""
    # Log the actual error for debugging (in production, send to logging service)
    print(f"Unexpected error: {exc}")
    
    # Return generic error to user (don't leak stack traces!)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."}
    )

# Configure CORS (Cross-Origin Resource Sharing)
# Restricts which domains can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # Only specific domains
    allow_credentials=False,            # No credentials for public endpoints
    allow_methods=["GET", "POST"],      # Only necessary HTTP methods
    allow_headers=["Content-Type", "X-API-Key"],  # Only necessary headers
)


# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)


# Root endpoint
@app.get("/", tags=["root"])
async def home():
    """API root endpoint."""
    return {"msg": "Customer Churn System"}


# Include all routers without prefix
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(ab_testing.router)
app.include_router(data.router)
app.include_router(evidently_router)
