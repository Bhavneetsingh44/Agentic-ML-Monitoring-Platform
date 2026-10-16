import pandas as pd

from src.models.predict import load_churn_model
from src.data.feature_engineering import create_features
from src.explainability.shap_explainer import explain_prediction


# Load trained MLflow model
model = load_churn_model()

# Create one example customer
customer = pd.DataFrame([{
    "Year": 2023,
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 125000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000.0
}])

# Create the same engineered features used during training
customer = create_features(customer)

# Get churn prediction
probability = model.predict_proba(customer)[:, 1][0]

print(f"\nChurn probability: {probability:.2%}")

# Explain prediction with SHAP
explanation = explain_prediction(model, customer)

print("\nTop reasons behind prediction:")
print(
    explanation[
        ["feature", "shap_value", "importance"]
    ].head(10)
)