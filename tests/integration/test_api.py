from fastapi.testclient import TestClient
from src.api.main import app

def test_predict_endpoint_works():
    client = TestClient(app)
    payload = {"Recency": 30, "Frequency": 2, "Monetary": 120.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "risk_score" in r.json()
