from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, RiskPrediction
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Credit Risk API", version="1.0")

# Load model from MLflow registry
model_uri = "models:/best_credit_model/production"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Credit Risk Model API is running"}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(features: CustomerFeatures):
    df = pd.DataFrame([features.dict()])
    score = model.predict(df)[0]
    return RiskPrediction(risk_score=score)
