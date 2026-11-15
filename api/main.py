from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.customer360.features import build_features
from src.customer360.retention import retention_recommendation, risk_label

app = FastAPI(title="Customer360 API", version="1.0.0",
              description="CLV, churn risk, and retention recommendation API")

class CustomerInput(BaseModel):
    recency_days: int = Field(ge=0, le=5000)
    frequency: int = Field(ge=0)
    monetary_value: float = Field(ge=0)
    tenure_months: int = Field(ge=1)
    avg_order_value: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    engagement_score: float = Field(ge=0, le=1)

def load_models():
    base = Path("artifacts")
    try:
        return (joblib.load(base / "clv_model.joblib"),
                joblib.load(base / "churn_model.joblib"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Train models first: python scripts/run_pipeline.py") from exc

@app.get("/")
def root():
    return {"message": "Customer360 API", "docs": "/docs"}

@app.post("/predict")
def predict(customer: CustomerInput):
    clv_model, churn_model = load_models()
    row = pd.DataFrame([customer.model_dump()])
    X = build_features(row)
    clv = max(0.0, float(clv_model.predict(X)[0]))
    probability = float(churn_model.predict_proba(X)[0, 1])
    strategy, action = retention_recommendation(clv, probability, customer.engagement_score)
    return {
        "predicted_clv": round(clv, 2),
        "churn_probability": round(probability, 4),
        "churn_risk": risk_label(probability),
        "retention_strategy": strategy,
        "recommended_action": action
    }
