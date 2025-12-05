"""Application lifespan management and model loading.

This module handles:
- Application startup (DVC sync, DB init, model loading)
- Background tasks (periodic model reloading)
- Application shutdown
"""

import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

from src.api.core.database import init_db
from src.api.core.state import AppState
from src.experiments.ab import ExperimentConfig


# Environment variables
ENV = os.getenv("ENV", "dev")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = logging.getLogger(__name__)


async def load_models(app: FastAPI, model_name: str = MODEL_NAME) -> None:
    """Load Production and Staging models from MLflow registry.
    
    Uses asyncio.to_thread() for all blocking MLflow operations to prevent
    blocking the FastAPI event loop during model loading.
    
    Args:
        app: FastAPI application instance
        model_name: Name of the model in MLflow registry
    """
    try:
        # Run blocking MLflow search in thread pool
        models = await asyncio.to_thread(
            mlflow.search_model_versions,
            filter_string=f"name='{model_name}'", 
            max_results=1000
        )
        
        for m in models:
            if m.current_stage == "Production":
                app.state.app_state.prod_version = m.version
                app.state.app_state.prod_source = m.source
            if m.current_stage == "Staging":
                app.state.app_state.stag_version = m.version
                app.state.app_state.stag_source = m.source

        logger.info(
            "Production model version: %s, source: %s",
            app.state.app_state.prod_version,
            app.state.app_state.prod_source
        )
        logger.info(
            "Staging model version: %s, source: %s",
            app.state.app_state.stag_version,
            app.state.app_state.stag_source
        )
        
        # Try to load sklearn models for fast inference (in thread pool)
        try:
            if app.state.app_state.prod_source:
                app.state.app_state.model_A = await asyncio.to_thread(
                    mlflow.sklearn.load_model,
                    app.state.app_state.prod_source
                )
                logger.info("Loaded Production model: %s", app.state.app_state.prod_source)
            
            if app.state.app_state.stag_source:
                app.state.app_state.model_B = await asyncio.to_thread(
                    mlflow.sklearn.load_model,
                    app.state.app_state.stag_source
                )
                logger.info("Loaded Staging model: %s", app.state.app_state.stag_source)
        except Exception as e:
            # Non-fatal: keep running without preloaded sklearn models
            app.state.app_state.model_A = None
            app.state.app_state.model_B = None
            logger.error("Error loading sklearn model(s); continuing without preload: %s", e, exc_info=True)
        
        # Preload PyFunc fallback model to avoid first-request latency (in thread pool)
        try:
            uri = f"models:/{model_name}/{MODEL_STAGE}"
            app.state.app_state.pyfunc_model = await asyncio.to_thread(
                mlflow.pyfunc.load_model, uri
            )
            logger.info("Preloaded PyFunc model from: %s", uri)
            
            # Get version number (in thread pool)
            try:
                client = MlflowClient()
                versions = await asyncio.to_thread(
                    client.get_latest_versions, model_name, stages=[MODEL_STAGE]
                )
                if versions:
                    app.state.app_state.pyfunc_model_version = versions[0].version
                    logger.info("PyFunc model version: %s", app.state.app_state.pyfunc_model_version)
            except Exception:
                pass
        except Exception as e:
            logger.warning("PyFunc model preload failed: %s", e)
            
    except Exception as e:
        logger.error("Error loading models from registry: %s", e, exc_info=True)


async def model_reloader(app: FastAPI, interval: int = 300) -> None:
    """Background task to periodically reload models from MLflow.
    
    Args:
        app: FastAPI application instance
        interval: Reload interval in seconds (default: 300 = 5 minutes)
    """
    await asyncio.sleep(5)  # Initial delay
    
    while True:
        try:
            await load_models(app, MODEL_NAME)
        except Exception as e:
            logger.error("Error during periodic model reload: %s", e, exc_info=True)
        
        await asyncio.sleep(interval)


def sync_dvc_data() -> None:
    """Synchronize DVC-tracked data (non-blocking on failure)."""
    try:
        logger.info("Pulling DVC data...")
        subprocess.run(["dvc", "pull", "-v"], check=True)
        logger.info("DVC data synchronized")
    except Exception as e:
        logger.error("DVC pull failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager.
    
    Handles:
    - Database initialization
    - DVC data synchronization
    - AppState initialization
    - Model loading
    - Background task setup
    
    Args:
        app: FastAPI application instance
    """
    # Initialize database (skip in test environment)
    if ENV != "test":
        init_db()

    # Sync DVC data
    sync_dvc_data()

    # Initialize application state
    app.state.app_state = AppState()
    
    # Initialize A/B testing configuration
    ab_enabled_env = os.getenv("AB_ENABLED", "true").lower() in {"1", "true", "yes"}
    app.state.app_state.ab_config = ExperimentConfig(enabled=ab_enabled_env)

    # Load models (non-fatal on failure)
    try:
        await load_models(app, MODEL_NAME)
    except Exception as e:
        logger.warning("Initial model preload skipped due to error: %s", e)
    
    # Start background model reloader task
    reloader_task = asyncio.create_task(model_reloader(app, interval=300))

    try:
        yield  # Application runs here
    finally:
        # Cleanup on shutdown
        reloader_task.cancel()
        logger.info("Application shutdown complete")
