from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    # Add more fields as needed

class RiskPrediction(BaseModel):
    risk_score: float
