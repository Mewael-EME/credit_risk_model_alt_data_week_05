from fastapi.testclient import TestClient
from src.api.main import app

def test_predict_endpoint():
    client = TestClient(app)
    payload = {"Recency": 30, "Frequency": 2, "Monetary": 120.0}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "risk_score" in res.json()
