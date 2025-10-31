from dotenv import load_dotenv
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
os.environ["ENV"] = "test"

from fastapi.testclient import TestClient

load_dotenv()

key = os.getenv("API_KEY_SECRET")


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Customer Churn System"}


def test_check_health_without_key(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 403


# def test_check_health(client: TestClient):
#     headers = {"X-API-Key": key}
#     response = client.get("/health", headers=headers)
#     assert response.status_code == 200
#     assert response.json() == {"check": "I'm ok! No worry"}