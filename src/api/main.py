from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.core.security import verify_api_key
from src.api.routes import users,auth
from src.api.core.database import init_db
from pydantic import BaseModel
# from prometheus_client import Counter, Histogram, Gauge
# import time


app = FastAPI(title="Customer Churn Prediction")

Instrumentator().instrument(app).expose(app)


init_db()

@app.get("/")
async def home():
    return {"msg":"Customer Churn System"}


@app.get("/health", dependencies=[Depends(verify_api_key)])
async def check_healh():
    return {"check": "I'm ok! No worry"}


app.include_router(users.router)
app.include_router(auth.router)



# # Métriques personnalisées
# prediction_counter = Counter(
#     'churn_predictions_total',
#     'Total number of churn predictions',
#     ['model_version', 'prediction_result']
# )

# prediction_duration = Histogram(
#     'churn_prediction_duration_seconds',
#     'Time spent processing prediction'
# )

# active_users = Gauge(
#     'churn_api_active_users',
#     'Number of active users'
# )

# # Exemple d'utilisation dans vos endpoints
# async def predict_churn(data):
#     start_time = time.time()
    
#     # Votre logique de prédiction
#     result = your_model.predict(data)
    
#     # Enregistrer les métriques
#     prediction_counter.labels(
#         model_version='v1.0',
#         prediction_result='churn' if result == 1 else 'no_churn'
#     ).inc()
    
#     prediction_duration.observe(time.time() - start_time)
    
#     return result