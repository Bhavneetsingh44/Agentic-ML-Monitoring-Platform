import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
):
    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    return result

def get_drifted_features(report):
    result_dict = report.dump_dict()
    drifted_features = []

    for metric in result_dict["metric_results"].values():

        params = (
            metric.get("metric_value_location", {})
            .get("metric", {})
            .get("params", {})
        )

        if params.get("type") != "evidently:metric_v2:ValueDrift":
            continue

        feature = params.get("column")
        score = metric.get("value")
        threshold = params.get("threshold")

        if (
            feature is not None
            and score is not None
            and threshold is not None
            and score > threshold
        ):
            drifted_features.append(feature)

    return drifted_features

def create_drift_alert(report):
    drifted_features = get_drifted_features(report)

    alert = {
        "status": "WARNING" if drifted_features else "OK",
        "drifted_count": len(drifted_features),
        "drifted_features": drifted_features,
        "action": (
            "Investigate feature drift"
            if drifted_features
            else "No investigation required"
        )
    }

    return alert