from langchain_core.tools import tool
import pandas as pd


def compare_feature_distributions(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drifted_features: list[str],
) -> list[dict]:

    results = []

    for feature in drifted_features:

        if feature not in reference_data.columns:
            continue

        if feature not in current_data.columns:
            continue

        reference = reference_data[feature]
        current = current_data[feature]

        if not (
            pd.api.types.is_numeric_dtype(reference)
            and pd.api.types.is_numeric_dtype(current)
        ):
            continue

        old_mean = reference.mean()
        new_mean = current.mean()

        difference = new_mean - old_mean

        if old_mean != 0:
            percentage_change = (difference / abs(old_mean)) * 100
        else:
            percentage_change = None

        results.append({
            "feature": feature,
            "reference_mean": round(float(old_mean), 2),
            "current_mean": round(float(new_mean), 2),
            "difference": round(float(difference), 2),
            "percentage_change": (
                round(float(percentage_change), 2)
                if percentage_change is not None
                else None
            ),
        })

    return results


@tool
def investigate_feature_importance() -> str:
    """Investigate which features influence model predictions."""
    return "SHAP_INVESTIGATION"


@tool
def investigate_distribution_drift() -> str:
    """Investigate changes in input feature distributions."""
    return "DRIFT_INVESTIGATION"