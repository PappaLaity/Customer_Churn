from fastapi import Depends, FastAPI, HTTPException, requests
from fastapi.responses import JSONResponse
import os
import random
from fastapi.openapi.utils import get_openapi
import mlflow
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

mlflow.set_tracking_uri("http://mlflow:5000")

# model_A = mlflow.pyfunc.load_model("models:/CustomerChurnModel/Production")
# model_B = mlflow.pyfunc.load_model("models:/CustomerChurnModel/Staging")

churn = ["No", "Yes"]
ENV = os.getenv("ENV", "dev")
if ENV != "test":
    init_db()


@app.get("/")
async def home():
    return {"msg": "Customer Churn System"}


@app.get("/models")
async def get_models():
    models = []
    models = mlflow.search_model_versions(filter_string="name='CustomerChurnModel'", max_results=1000)
    return {
        "models": [
            {
                "version": m.version,
                "current_stage": m.current_stage,
                "creation_timestamp": m.creation_timestamp,
                "last_updated_timestamp": m.last_updated_timestamp,
            }
            for m in models
        ]
    }


@app.get("/customers/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    infos = []
    file_path = Path("Data/production/customer_production_data.csv")
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


@app.post("/survey/submit")
async def submit_survey(input: InputCustomer = None):

    # Data Validation
    data = input
    # Prepare Data for Prediction
    # Make Prediction
    # Prepare Input and Prediction
    prediction = predict_churn(data)
    # Store it in the production data

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
    # result = predict(data)
    time.sleep(5)
    # Enregistrer les métriques
    prediction_counter.labels(
        model_version="v1.0", prediction_result="churn" if result == 1 else "no_churn"
    ).inc()

    prediction_duration.observe(time.time() - start_time)

    return result


# @app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    model_choice = "A" if random.random() < 0.8 else "B"

    start = time.time()
    # preds = model_A.predict(df) if model_choice == "A" else model_B.predict(df)
    latency = time.time() - start

    # Log locally or send to MLflow for analysis
    mlflow.log_metric("latency", latency)
    mlflow.log_param("model_used", model_choice)

    return {
        "model": "Production" if model_choice == "A" else "Staging",
        # "prediction": preds.tolist(),
        "latency": latency,
    }
