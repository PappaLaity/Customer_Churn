import asyncio
import os
import random
import subprocess
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, requests
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
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


model_name = "CustomerChurnModel"

churn = ["No", "Yes"]
ENV = os.getenv("ENV", "dev")


@asynccontextmanager
async def lifespan(app: FastAPI):

    if ENV != "test":
        init_db()

    print("Pulling DVC data...")
    subprocess.run(["dvc", "pull", "-v"], check=True)
    print("DVC data synchronized.")

    app.state.model_A = None
    app.state.model_B = None
    app.state.stag_version = None
    app.state.stag_source = None
    app.state.prod_version = None
    app.state.prod_source = None
    """Gestionnaire de cycle de vie (démarrage + arrêt)."""
    # --- Phase de démarrage ---
    load_model(model_name)
    print("Modèle initial chargé.")
    task = asyncio.create_task(model_reloader(interval=300))

    yield

    # --- Phase d’arrêt ---
    task.cancel()
    print("Arrêt du rechargement périodique du modèle.")


# Crée ton application FastAPI avec lifespan
app = FastAPI(description="Customer Churn Prediction", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Production_Customer_Churn_API")


@app.get("/")
async def home():
    return {"msg": "Customer Churn System"}


@app.get("/model/version", dependencies=[Depends(verify_api_key)])
async def get_model_version():
    return {
        "production_model_version": app.state.prod_version,
        "staging_model_version": app.state.stag_version,
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
async def submit_survey(input: InputCustomer, background_tasks: BackgroundTasks):

    file_path = Path("data/production/production.csv")
    # Data Validation
    data = input
    result = await predict_churn(data)
    # Make Prediction
    customer_data = input.model_dump(by_alias=True)
    customer_data["Churn"] = int(result)
    df = pd.DataFrame([customer_data])
    df["Churn"] = result

    csv_columns = [
        "tenure",
        "InternetService_Fiber_optic",
        "Contract_Two_year",
        "PaymentMethod_Electronic_check",
        "No_internet_service",
        "TotalCharges",
        "MonthlyCharges",
        "PaperlessBilling",
        "Churn",
    ]

    # Vérifier si le fichier existe ET non vide
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=csv_columns)
    else:
        df_existing = pd.DataFrame(columns=csv_columns)

    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv(file_path, index=False)
    # Store it in the production data
    background_tasks.add_task(dvc_push_background)
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

    results = await predict(data)
    result = results["prediction"]
    # Enregistrer les métriques
    prediction_counter.labels(
        model_version="v1.0", prediction_result="churn" if result == 1 else "no_churn"
    ).inc()

    prediction_duration.observe(time.time() - start_time)

    return result


async def predict(data: InputCustomer):

    df = pd.DataFrame([data.model_dump()])
    if app.state.model_A and app.state.model_B:
        model_choice = "A" if random.random() < 0.8 else "B"
    else:
        model_choice = "A"

    start = time.time()
    preds = (
        app.state.model_A.predict(df)
        if model_choice == "A"
        else app.state.model_B.predict(df)
    )
    # result = model_A.predict(df)[0]
    print(f"Predicted result: {preds[0]}")
    latency = time.time() - start

    # Log locally or send to MLflow for analysis
    mlflow.log_metric("latency", latency)
    mlflow.log_param("model_used", model_choice)

    return {
        "model": "Production" if model_choice == "A" else "Staging",
        "prediction": preds[0],
        "latency": latency,
    }


# No more needed

# @app.post("/predict/test", dependencies=[Depends(verify_api_key)])
# async def predict_churn_test(sample: InputCustomer):

#     # sample = {
#     #     "Contract": 0,
#     #     "tenure": 1,
#     #     "OnlineSecurity": 0,
#     #     "TechSupport": 0,
#     #     "TotalCharges": 29.85,
#     #     "OnlineBackup": 2,
#     #     "MonthlyCharges": 29.85,
#     #     "PaperlessBilling": 1,
#     # }


#     df = pd.DataFrame([sample.model_dump()])

#     # Faire la prédiction
#     result = model_A.predict(df)
#     prediction = model_A.predict(df)[0]
#     probability = model_A.predict_proba(df)[0][1]

#     print(f"Prediction: {churn[prediction]}, Probability of Churn: {probability:.4f}, Result: {result}")

#     return {"prediction": int(prediction), "probability": round(float(probability), 4)}
#     # return {"prediction": prediction, "probability": probability, "input": sample}


def load_model(model_name="CustomerChurnModel"):

    models = mlflow.search_model_versions(
        filter_string="name='CustomerChurnModel'", max_results=1000
    )
    for m in models:
        if m.current_stage == "Production":
            app.state.prod_version = m.version
            app.state.prod_source = m.source
        if m.current_stage == "Staging":
            app.state.stag_version = m.version
            app.state.stag_source = m.source

    print(
        f"Production model version: {app.state.prod_version}, source: {app.state.prod_source}"
    )
    print(
        f"Staging model version: {app.state.stag_version}, source: {app.state.stag_source}"
    )
    try:
        if app.state.prod_source:
            app.state.model_A = mlflow.sklearn.load_model(app.state.prod_source)
            print(f"Loaded Production model version: {app.state.prod_source}")
        if app.state.stag_source:
            app.state.model_B = mlflow.sklearn.load_model(app.state.stag_source)
            print(f"Loaded Production model version: {app.state.stag_source}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    # return model_A, model_B, prod_version, stag_version,prod_source,stag_source


async def model_reloader(interval: int = 300):
    """Tâche asynchrone qui recharge le modèle toutes les 'interval' secondes (300s = 5min)."""
    await asyncio.sleep(5)
    while True:
        try:
            load_model(model_name)
        except Exception as e:
            print(f"Erreur lors du rechargement périodique: {e}")
        await asyncio.sleep(interval)


async def dvc_push_background():
    """Exécute un `dvc push` sans bloquer le thread principal."""
    process = await asyncio.create_subprocess_exec(
        "dvc",
        "push",
        "-v",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print("[DVC] Push successful:\n", stdout.decode())
    else:
        print("[DVC] Push failed:\n", stderr.decode())
