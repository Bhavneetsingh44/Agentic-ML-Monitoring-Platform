
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_homepage_returns_200():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Bank Churn API is running"
    }

def test_predict_rejects_invalid_credit_score():
    invalid_customer = {
        "Year": 2020,
        "CreditScore": 50,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 5,
        "Balance": 120000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000
    }

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422



def test_drift_endpoint_returns_valid_response():
    response = client.get("/monitor/drift")

    assert response.status_code == 200

    result = response.json()
    assert "status" in result
    assert "drifted_count" in result
    assert "drifted_features" in result
    assert isinstance(result["drifted_count"], int)
    assert isinstance(result["drifted_features"], list)
