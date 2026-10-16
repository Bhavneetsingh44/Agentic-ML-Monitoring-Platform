import pandas as pd

from src.data.feature_engineering import create_features
from src.monitoring.drift import (
    detect_data_drift,
    create_drift_alert,
)
from src.agents.monitoring_agent import graph


def run_monitoring_agent(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
):

    # Detect drift
    report = detect_data_drift(reference_data, current_data)
    alert = create_drift_alert(report)

    # Initialize LangGraph state
    initial_state = {
        "status": alert["status"],
        "drifted_features": alert["drifted_features"],
        "recommendation": "",
        "important_features": [],
        "reference_data": reference_data,
        "customer_data": current_data,
        "llm_analysis": "",
        "selected_tool": "",
        "tool_result": "",
        "completed_tools": [],
        "investigation_results": [],
    }

    # Run agent
    result = graph.invoke(initial_state)

    return {
        "status": result["status"],
        "drifted_features": result["drifted_features"],
        "completed_tools": result["completed_tools"],
        "important_features": result["important_features"],
        "recommendation": result["recommendation"],
        "llm_analysis": result["llm_analysis"],
        "investigation_results": result["investigation_results"],
    }