import pandas as pd

from src.data.feature_engineering import create_features
from src.monitoring.drift import detect_data_drift, create_drift_alert
from src.agents.monitoring_agent import graph


# 1. Load data
df = pd.read_csv("data/raw/European_Bank.csv")

df = df.drop(columns=["CustomerId", "Surname", "Exited"])


# 2. Prepare reference and current data
reference_data = create_features(df.iloc[:8000].copy())

current_raw = df.iloc[8000:].copy()

# Simulate data drift
current_raw["Age"] += 15
current_raw["Balance"] *= 1.5

current_data = create_features(current_raw)


# 3. Detect drift with Evidently
report = detect_data_drift(reference_data, current_data)

alert = create_drift_alert(report)

print("Evidently Alert:", alert)


# 4. Run LangGraph agent
result = graph.invoke({
    "status": alert["status"],
     "reference_data": reference_data,
    "drifted_features": alert["drifted_features"],
    "recommendation": "",
    "important_features": [],
    "completed_tools": [],
    "investigation_results": [],
    "customer_data": current_data,
    "selected_tool": "",
    "tool_result": "",
    "llm_analysis": ""

})


# 5. Display results
print("Agent Recommendation:", result["recommendation"])
print("Important SHAP Features:", result["important_features"])

print("\nSelected Tool:", result.get("selected_tool"))
print("\nTool Result:", result.get("tool_result"))

print("\nLLM Investigation Report:")
print(result["llm_analysis"])

print("\nCompleted Tools:", result["completed_tools"])

print("\nAll Investigation Results:")
for item in result["investigation_results"]:
    print(item)