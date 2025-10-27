import random
from fastapi import Depends, FastAPI, HTTPException, requests
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.entities.customerInput import InputCustomer
from src.api.core.security import verify_api_key
from src.api.routes import users, auth
from src.api.core.database import init_db
from pathlib import Path
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, Gauge
import time


app = FastAPI(title="Customer Churn Prediction")

Instrumentator().instrument(app).expose(app)

churn = ["No", "Yes"]
init_db()


@app.get("/")
async def home():
    return {"msg": "Customer Churn System"}


@app.get("/customers/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    infos = []
    file_path = Path("Data/production/customer_production_data.csv")

    # if Production Data Exist load data
    #   - Data/production/customer_production_data.csv
    #   - load it from bottom to top
    # else
    #   - return empty table

    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df) == 0:
                return JSONResponse(
                    content={"status": "success", "data": [], "count": 0}
                )

            df_reversed = df.iloc[::-1].reset_index(drop=True)
            return JSONResponse(
                content={
                    "status": "success",
                    "columns": df_reversed.columns.tolist(),
                    "data": df_reversed.values.tolist(),
                    "count": len(df_reversed),
                }
            )
        except pd.errors.EmptyDataError:
            # Fichier complètement vide (pas même d'en-têtes)
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
                status_code=500, detail=f"Error reading production data: {str(e)}"
            )
    else:
        return JSONResponse(
            content={
                "status": "success",
                "data": [],
                "count": 0,
                "message": "No production data found",
            }
        )


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def check_healh():
    return {"check": "I'm ok! No worry"}


# async def predict_churn(data):
#     prediction = 1
#     pass


@app.post("/survey/submit")
async def submit_survey(input: InputCustomer = None):

    # Data Validation
    # Prepare Data for Prediction
    # Make Prediction
    # Prepare Input and Prediction
    # Store it in the production data
    prediction = predict_churn()

    return {"success": "Thanky you for your submission"}


app.include_router(users.router)
app.include_router(auth.router)


# # Métriques personnalisées
prediction_counter = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["model_version", "prediction_result"],
)

prediction_duration = Histogram(
    "churn_prediction_duration_seconds", "Time spent processing prediction"
)

active_users = Gauge("churn_api_active_users", "Number of active users")


# # Exemple d'utilisation dans vos endpoints
async def predict_churn(data=None):
    start_time = time.time()

    # Votre logique de prédiction
    # result = your_model.predict(data)
    result = random.randint(0, 1)
    time.sleep(5)
    # Enregistrer les métriques
    prediction_counter.labels(
        model_version="v1.0", prediction_result="churn" if result == 1 else "no_churn"
    ).inc()

    prediction_duration.observe(time.time() - start_time)

    return result
