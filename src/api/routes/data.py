"""Data management endpoints.

This router handles:
- Customer production data retrieval
- DVC data versioning
"""

import asyncio
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.core.security import verify_api_key


router = APIRouter(tags=["data"])

logger = logging.getLogger(__name__)


@router.get("/customers/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    """Get production customer data.
    
    Returns customer data from the production CSV file in reverse chronological order.
    
    Returns:
        JSON response with customer data and metadata
        
    Raises:
        HTTPException: If there's an error reading the file
    """
    file_path = Path("data/production/production.csv")
    
    if not file_path.exists():
        return JSONResponse(
            content={
                "status": "success",
                "data": [],
                "count": 0,
                "message": "No production data found",
            }
        )
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            return JSONResponse(
                content={"status": "success", "data": [], "count": 0}
            )

        # Reverse order to show most recent first
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        data = df_reversed.to_dict(orient="records")
        
        return JSONResponse(
            content={
                "status": "success",
                "columns": df_reversed.columns.tolist(),
                "data": data,
                "count": len(df_reversed),
            }
        )
        
    except pd.errors.EmptyDataError:
        return JSONResponse(
            content={
                "status": "success",
                "data": [],
                "count": 0,
                "message": "Production data file is empty (no headers)",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading production data: {str(e)}"
        )


async def dvc_push_background():
    """Background task to push data changes to DVC remote.
    
    Runs asynchronously without blocking the API response.
    """
    process = await asyncio.create_subprocess_exec(
        "dvc",
        "push",
        "-v",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        logger.info("DVC push successful: %s", stdout.decode())
    else:
        logger.error("DVC push failed: %s", stderr.decode())
