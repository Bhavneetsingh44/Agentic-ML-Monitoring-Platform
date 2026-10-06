from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.models.predict import load_churn_model

app = FastAPI()

# Load model once when API starts
model = load_churn_model()

THRESHOLD = 0.3


class Customer(BaseModel):
    Year: int
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    return {"message": "Bank Churn API is running"}


@app.post("/predict")
def predict(customer: Customer):

    # Convert incoming customer data to DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Create the same engineered features used during training
    data["BalanceToSalary"] = (
        data["Balance"] / (data["EstimatedSalary"] + 1)
    )

    data["ProductsPerYear"] = (
        data["NumOfProducts"] / data["Age"]
    )

    data["EngagementScore"] = (
        data["IsActiveMember"] + data["HasCrCard"]
    )

    data["AgeWhenJoined"] = (
        data["Age"] - data["Tenure"]
    )

    # Get churn probability
    probability = model.predict_proba(data)[:, 1][0]

    # Apply our selected threshold
    prediction = int(probability >= THRESHOLD)

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "churn": bool(prediction),
        "threshold": THRESHOLD
    }