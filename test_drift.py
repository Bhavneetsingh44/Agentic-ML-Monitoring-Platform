import pandas as pd

from src.data.feature_engineering import create_features
from src.monitoring.drift import detect_data_drift, get_drifted_features
from src.monitoring.drift import (
    detect_data_drift,
    get_drifted_features,
    create_drift_alert
)

# 1. Load real bank data
df = pd.read_csv("data/raw/European_Bank.csv")


# 2. Remove columns that are not model inputs
df = df.drop(columns=[
    "CustomerId",
    "Surname",
    "Exited"
])


# 3. Split RAW data first
reference_raw = df.iloc[:8000].copy()
current_raw = df.iloc[8000:].copy()


# 4. Simulate changes in incoming production data
current_raw["Age"] = current_raw["Age"] + 15
current_raw["Balance"] = current_raw["Balance"] * 1.5
current_raw["EstimatedSalary"] = current_raw["EstimatedSalary"] * 1.3


# 5. NOW perform feature engineering
reference_data = create_features(reference_raw)
current_data = create_features(current_raw)


# 6. Run drift monitoring
report = detect_data_drift(
    reference_data=reference_data,
    current_data=current_data
)


# 7. Save report
report.save_html("real_drift_report.html")

drifted_features = get_drifted_features(report)



print("\n--- DRIFT ALERT ---")
print("Drifted features:", len(drifted_features))

for feature in drifted_features:
    print("🚨", feature)

if len(drifted_features) > 0:
    print("\n⚠️ WARNING: Feature drift detected!")
    print("Action: Investigate the affected features.")
else:
    print("\n✅ No feature drift detected.")

print("Reference shape:", reference_data.shape)
print("Current shape:", current_data.shape)
print("Features monitored:", len(reference_data.columns))
print("Drift report created: real_drift_report.html")

print("\nReport result type:")
print(type(report))

result_dict = report.dump_dict()

drifted_features = get_drifted_features(report)

first_metric = next(iter(result_dict["metric_results"].values()))

drifted_count = int(first_metric["count"]["value"])
drift_share = first_metric["share"]["value"]

print("\n--- DRIFT MONITORING ---")
print("Drifted features:", drifted_count)
print("Drift share:", round(drift_share * 100, 2), "%")

alert = create_drift_alert(report)

print("\n--- STRUCTURED DRIFT ALERT ---")
print(alert)


