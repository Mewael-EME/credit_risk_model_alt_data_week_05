from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, RiskPrediction
import mlflow.pyfunc, pandas as pd, os

app = FastAPI(title="Credit Risk API", version="1.0")

MODEL_URI = os.getenv("MODEL_URI", "models:/best_credit_model/production")
model = None
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception:
    pass  # allow API to start; handle fallback in endpoint

@app.get("/")
def root():
    return {"message": "Credit Risk Model API up", "model_loaded": model is not None}

@app.post("/predict", response_model=RiskPrediction)
def predict(features: CustomerFeatures):
    df = pd.DataFrame([features.dict()])
    if model is None:
        # Fallback heuristic if no registry available
        score = float(0.6*(df.Recency.values[0] > 30) + 0.3*(df.Frequency.values[0] < 2) + 0.1*(df.Monetary.values[0] < 100))
        return RiskPrediction(risk_score=min(0.99, max(0.01, score)))
    score = float(model.predict(df)[0])
    return RiskPrediction(risk_score=score)
