import mlflow.sklearn


def load_churn_model():
    model_uri = "models:/m-c5a476234c37498c93061a724ed1db5f"

    model = mlflow.sklearn.load_model(model_uri)

    return model