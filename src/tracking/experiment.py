import mlflow
import mlflow.sklearn


def log_experiment(model, metrics, threshold):

    mlflow.set_experiment("Bank_Churn_Monitoring")

    with mlflow.start_run():

        # Save model settings
        mlflow.log_param("algorithm", "GradientBoosting")
        mlflow.log_param("threshold", threshold)

        # Save validation metrics
        mlflow.log_metrics(metrics)

        # Save trained pipeline
        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("Experiment saved successfully!")