import asyncio
import os
import random
from fastapi import Depends, FastAPI, HTTPException, requests
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
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

model_name = "CustomerChurnModel"



def load_model(model_name="CustomerChurnModel"):
    model_A = None
    model_B = None
    stag_version = None
    stag_source = None
    prod_version = None
    prod_source = None
    models = mlflow.search_model_versions(
        filter_string="name='CustomerChurnModel'", max_results=1000
    )
    for m in models:
        if m.current_stage == "Production":
            prod_version = m.version
            prod_source = m.source
        if m.current_stage == "Staging":
            stag_version = m.version
            stag_source = m.source

    print(f"Production model version: {prod_version}, source: {prod_source}")
    print(f"Staging model version: {stag_version}, source: {stag_source}")
    try:
        if prod_source:
            model_A = mlflow.sklearn.load_model(prod_source)
            print(f"Loaded Production model version: {prod_source}")
        if stag_source:
            model_B = mlflow.sklearn.load_model(stag_source)
            print(f"Loaded Production model version: {stag_source}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model_A, model_B, prod_version, stag_version,prod_source,stag_source

model_A, model_B, prod_version, stag_version,prod_source,stag_source = load_model(model_name)

@app.post("/reload")
def manual_reload():
    load_model(model_name)
    return {"message": f"Modèle version {prod_version} rechargé manuellement"}


# async def model_reloader(interval: int = 300):
#     """Tâche asynchrone qui recharge le modèle toutes les 'interval' secondes (300s = 5min)."""
#     await asyncio.sleep(5)
#     while True:
#         try:
#             load_model(model_name)
#         except Exception as e:
#             print(f"Erreur lors du rechargement périodique: {e}")
#         await asyncio.sleep(interval)


# @app.on_event("startup")
# async def startup_event():
    """Au démarrage, charge le modèle et lance la boucle de rechargement."""
    load_model(model_name)
    asyncio.create_task(model_reloader(interval=300))  # 300 secondes = 5 min



churn = ["No", "Yes"]
ENV = os.getenv("ENV", "dev")
if ENV != "test":
    init_db()


@app.get("/")
async def home():
    return {"msg": "Customer Churn System"}


@app.get("/model/version", dependencies=[Depends(verify_api_key)])
async def get_model_version():
    return {
        "production_model_version": prod_version,
        "staging_model_version": stag_version,
    }


@app.get("/models")
async def get_models():
    models = []
    models = mlflow.search_model_versions(
        filter_string="name='CustomerChurnModel'", max_results=1000
    )
    return {
        "models": [
            {
                "version": m.version,
                "current_stage": m.current_stage,
                "creation_timestamp": m.creation_timestamp,
                "last_updated_timestamp": m.last_updated_timestamp,
                "source": m.source,
                "run_id": m.run_id,
            }
            for m in models
        ]
    }


@app.get("/customers/infos", dependencies=[Depends(verify_api_key)])
async def get_customers_infos():
    infos = []
    file_path = Path("data/production/production.csv")
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


@app.post("/predict/test", dependencies=[Depends(verify_api_key)])
async def predict_churn_test(sample: InputCustomer):

    # sample = {
    #     "Contract": 0,
    #     "tenure": 1,
    #     "OnlineSecurity": 0,
    #     "TechSupport": 0,
    #     "TotalCharges": 29.85,
    #     "OnlineBackup": 2,
    #     "MonthlyCharges": 29.85,
    #     "PaperlessBilling": 1,
    # }

    # Charger le modèle

    # Si input_data est un dict, on le convertit en DataFrame
    # if isinstance(sample, dict):
    #     df = pd.DataFrame([sample])
    # else:
    #     df = sample.copy()

    df = pd.DataFrame([sample.model_dump()])

    # S'assurer que les colonnes sont dans le même ordre que celles du modèle
    # (tu peux adapter en fonction du preprocessing)

    # Faire la prédiction
    result = model_A.predict(df)
    prediction = model_A.predict(df)[0]
    probability = model_A.predict_proba(df)[0][1]

    print(f"Prediction: {churn[prediction]}, Probability of Churn: {probability:.4f}, Result: {result}")

    return {"prediction": int(prediction), "probability": round(float(probability), 4)}
    # return {"prediction": prediction, "probability": probability, "input": sample}
