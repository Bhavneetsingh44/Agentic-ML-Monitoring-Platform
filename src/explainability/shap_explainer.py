import pandas as pd
import shap


def explain_prediction(model, customer_data: pd.DataFrame):
    # Get preprocessing and ML model from our Pipeline
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    # Transform customer data exactly like model training data
    transformed_data = preprocessor.transform(customer_data)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(classifier)

    # Calculate SHAP values
    shap_values = explainer(transformed_data)

    # Build readable feature explanation
    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": transformed_data[0],
        "shap_value": shap_values.values[0]
    })

    # Most influential features first
    explanation["importance"] = explanation["shap_value"].abs()

    return explanation.sort_values(
        "importance",
        ascending=False
    )