import os

import pytest
from fastapi.testclient import TestClient

os.environ["ENV"] = "test"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"
os.environ.setdefault("API_KEY_SECRET", "my-default-api-key")

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _headers():
    return {"X-API-Key": os.getenv("API_KEY_SECRET", "my-default-api-key")}


def test_health_with_key(client):
    r = client.get("/health", headers=_headers())
    assert r.status_code == 200
    assert "check" in r.json()


def test_model_version_endpoint(client):
    r = client.get("/model/version", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert "production_model_version" in body
    assert "staging_model_version" in body


def test_baseline_set_and_get(client):
    payload = {"numeric": {"MonthlyCharges": [10.0, 20.0, 30.0]}}
    r1 = client.post("/monitoring/baseline", json=payload, headers=_headers())
    assert r1.status_code == 200
    r2 = client.get("/monitoring/baseline", headers=_headers())
    assert r2.status_code == 200
    assert "features" in r2.json()


def test_predict_dummy_model(client):
    payload = {
        "instances": [
            {
                "tenure": 12,
                "MonthlyCharges": 85.25,
                "TotalCharges": 1500.5,
                "PaperlessBilling": 1,
            }
        ],
        "return_proba": False,
    }
    r = client.post("/predict", json=payload, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1


def test_customers_infos_when_missing_file(client):
    # Ensure file does not exist so endpoint returns success with no data
    try:
        import pathlib

        p = pathlib.Path("data/production/production.csv")
        if p.exists():
            p.unlink()
    except Exception:
        pass
    r = client.get("/customers/infos", headers=_headers())
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "success"
