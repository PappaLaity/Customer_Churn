"""
Routes pour récupérer les données clients en production
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.core.config import settings
from src.api.core.security import verify_api_key

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get("/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    """
    Récupère les données clients en production

    Returns:
        Liste des clients avec leurs prédictions (ordre inversé, plus récents d'abord)
    """
    file_path = Path(settings.DATA_PATH)

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
            return JSONResponse(content={"status": "success", "data": [], "count": 0})

        # Inverser l'ordre (plus récents en premier)
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
        raise HTTPException(status_code=500, detail=f"Error reading production data: {str(e)}")
