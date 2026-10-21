from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from src.data.feature_engineering import create_features
from src.explainability.shap_explainer import explain_prediction
from fastapi import HTTPException

from src.agents.agent_service import run_monitoring_agent
from src.data.feature_engineering import create_features

from src.models.predict import load_churn_model

from src.monitoring.drift import (
    detect_data_drift,
    create_drift_alert
)
from fastapi import HTTPException

app = FastAPI()

# Load model once when API starts
model = load_churn_model()

THRESHOLD = 0.3


class Customer(BaseModel):
    Year: int = Field(ge=1900, le=2100)
    CreditScore: int = Field(ge=300, le=850)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=50)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Bank Churn API is running"}


@app.post("/predict")
def predict(customer: Customer):

    # Convert API input into DataFrame
    data = pd.DataFrame([customer.model_dump()])

    data = create_features(data)

    print("\n===== PREDICTION DEBUG =====", flush=True)
    print("Columns:", data.columns.tolist(), flush=True)
    print("Values:", data.to_dict(orient="records"), flush=True)
    print("Missing:", data.isna().sum().to_dict(), flush=True)

    if data.isna().any().any():
        raise HTTPException(
            status_code=422,
            detail="Invalid input: feature engineering produced missing values."
        )

    print("Input data:", data.to_dict(orient="records"), flush=True)
    print("Missing values:", data.isna().sum().to_dict(), flush=True)

    # Predict churn probability
    probability = model.predict_proba(data)[:, 1][0]

    # Apply our selected threshold
    prediction = int(probability >= THRESHOLD)

    # Explain prediction using SHAP
    explanation = explain_prediction(model, data)

    # Get top 5 most influential features
    top_factors = []

    for _, row in explanation.head(5).iterrows():
        top_factors.append({
            "feature": row["feature"],
            "shap_value": round(float(row["shap_value"]), 4),
            "direction": (
                "increases_churn"
                if row["shap_value"] > 0
                else "decreases_churn"
            )
        })

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "churn": bool(prediction),
        "threshold": THRESHOLD,
        "top_factors": top_factors
    }

@app.get("/monitor/drift")
def monitor_drift():

    df = pd.read_csv("data/raw/European_Bank.csv")
    df = df.drop(columns=["CustomerId", "Surname", "Exited"])

    reference_raw = df.iloc[:8000].copy()
    current_raw = df.iloc[8000:].copy()



    reference_data = create_features(reference_raw)
    current_data = create_features(current_raw)

    report = detect_data_drift(
        reference_data,
        current_data
    )

    alert = create_drift_alert(report)

    return alert

@app.post("/agent/investigate")
def investigate_agent():

    try:
        # Load sample dataset
        df = pd.read_csv("data/raw/European_Bank.csv")

        # Remove identifiers and target
        df = df.drop(
            columns=["CustomerId", "Surname", "Exited"]
        )

        # Historical reference data
        reference_raw = df.iloc[:8000].copy()

        # Current data (simulated drift)
        current_raw = df.iloc[8000:].copy()

        current_raw["Age"] += 15
        current_raw["Balance"] *= 1.5

        # Feature engineering
        reference_data = create_features(reference_raw)
        current_data = create_features(current_raw)

        # Run the agent
        result = run_monitoring_agent(
            reference_data=reference_data,
            current_data=current_data,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Agent investigation failed. Check server logs."
        ) from error