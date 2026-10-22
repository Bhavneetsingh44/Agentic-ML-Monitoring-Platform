import numpy as np
import pandas as pd

from src.monitoring.drift import detect_data_drift, create_drift_alert


def test_detects_deliberate_feature_drift():
    rng = np.random.default_rng(42)
    size = 500

    reference = pd.DataFrame({
        "Age": rng.normal(40, 5, size),
        "Balance": rng.normal(50000, 10000, size),
    })

    current = pd.DataFrame({
        "Age": rng.normal(65, 5, size),
        "Balance": rng.normal(100000, 10000, size),
    })

    report = detect_data_drift(reference, current)
    alert = create_drift_alert(report)

    assert alert["status"] == "WARNING"
    assert "Age" in alert["drifted_features"]
    assert "Balance" in alert["drifted_features"]
    assert alert["drifted_count"] == len(alert["drifted_features"])
