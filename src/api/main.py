from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.core.security import verify_api_key
from src.api.routes import users,auth
from src.api.core.database import init_db
from pydantic import BaseModel


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

# Expose Prometheus metrics
Instrumentator().instrument(app).expose(app)


# security = HTTPBearer(auto_error=False)

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Customer Churn API",
#         version="1.0.0",
#         description="API sécurisée avec JWT Bearer Token",
#         routes=app.routes,
#     )
#     openapi_schema["components"]["securitySchemes"] = {
#         "BearerAuth": {
#             "type": "http",
#             "scheme": "bearer",
#             "bearerFormat": "JWT"
#         }
#     }
#     for path in openapi_schema["paths"].values():
#         for method in path.values():
#             method.setdefault("security", [{"BearerAuth": []}])
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi
