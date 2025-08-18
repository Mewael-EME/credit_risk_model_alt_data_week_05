from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

class RiskPrediction(BaseModel):
    risk_score: float
