from typing import TypedDict
import pandas as pd

from langgraph.graph import StateGraph, START, END
from src.models.predict import load_churn_model
from src.explainability.shap_explainer import explain_prediction
from src.agents.llm_reasoner import get_llm
from src.agents.investigation_tools import compare_feature_distributions

from src.agents.investigation_tools import (
    investigate_feature_importance,
    investigate_distribution_drift,
)


class MonitoringState(TypedDict):
    status: str
    drifted_features: list[str]
    recommendation: str
    selected_tool: str
    tool_result: str
    reference_data: pd.DataFrame
    completed_tools: list[str]
    investigation_results: list[str]

    important_features: list[str]
    customer_data: pd.DataFrame

    llm_analysis: str

def check_alert(state: MonitoringState):
    if state["status"] == "WARNING":
        return {"recommendation": "Drift detected. Investigation required."}

    return {"recommendation": "No drift detected. Continue monitoring."}

def investigate_drift(state: MonitoringState):

    #  Load model from MLflow
    model = load_churn_model()

    #  Select up to 100 customers
    customers = state["customer_data"].sample(
        n=min(100, len(state["customer_data"])),
        random_state=42
    )

    #  Calculate SHAP for each customer
    shap_results = []

    for i in range(len(customers)):
        customer = customers.iloc[[i]]

        explanation = explain_prediction(model, customer)

        shap_results.append(
            explanation[["feature", "shap_value"]]
        )

    #  Combine all explanations
    combined = pd.concat(shap_results)

    #  Calculate average absolute SHAP importance
    combined["abs_shap"] = combined["shap_value"].abs()

    importance = (
        combined.groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=False)
    )

    #  Select top 5 influential features
    important_features = importance.head(5).index.tolist()

    #  Find overlap with drifting features
    drifted_features = state["drifted_features"]

    overlapping_features = [
        feature for feature in drifted_features
        if any(
            important == feature or important.endswith("__" + feature)
            for important in important_features
        )
    ]

    #  Return recommendation
    return {
        "important_features": important_features,
        "recommendation": (
            f"Prioritize investigation of: {', '.join(overlapping_features)}"
            if overlapping_features
            else "Drift detected, but no overlap with top 5 SHAP features."
        )
    }

def analyze_with_llm(state: MonitoringState):

    llm = get_llm()

    prompt = f"""
    You are an ML monitoring assistant.

    Evidently detected drift in:
    {state["drifted_features"]}

    Top 5 influential SHAP features:
    {state["important_features"]}

    Existing investigation recommendation:
    {state["recommendation"]}
    
    Investigation tool selected:
{state["selected_tool"]}

Investigation evidence:
{state["tool_result"]}

All completed investigations:
{state["completed_tools"]}

All collected evidence:
{state["investigation_results"]}

    Explain:
    1. Which drifting features deserve attention and why.
    2. What risks this creates for model predictions.
    3. What the ML engineer should investigate next.

    Rules:
    - Use simple English.
    - Keep your answer under 150 words.
    - Do not claim model accuracy has dropped without evidence.
    - Do not recommend automatic retraining.
    - Do not claim that model performance has declined
      unless actual performance metrics support it.
     - Recommend investigation and validation before retraining.
     - Do not invent SHAP results if SHAP was not executed.
     """
    """
    Business context:
    - This model predicts bank customer churn (Exited).
    - It does NOT predict loan default or credit risk.
    - Use only the investigation evidence provided.
    - Do not invent SHAP results if SHAP was not executed.
    - Data drift does not automatically mean accuracy has dropped.
    - Recommend retraining only after evaluating performance
      on recent labeled data.
    """
    """
    The model predicts customer churn only, not credit default.

    Use the exact numeric values supplied in the tool result.
    Do not invent or incorrectly round percentages.

    Do not assume that drift has reduced model performance.
    Do not claim that model predictions are biased without evidence.

    Prioritize investigation based on both:
    1. Magnitude of distribution change.
    2. Model feature importance, if SHAP results are available.

    If SHAP was not executed, explicitly state that
    feature importance was not evaluated in this run.
    """

    response = llm.invoke(prompt)

    return {
        "llm_analysis": response.content
    }

def route_alert(state: MonitoringState):

    if state["status"] == "WARNING":
        return "investigate_drift"

    return END

def select_investigation_tool(state: MonitoringState):

    available_tools = [
        tool for tool in [
            investigate_feature_importance,
            investigate_distribution_drift,
        ]
        if tool.name not in state["completed_tools"]
    ]

    if not available_tools:
        return {"selected_tool": "FINISH"}

    llm = get_llm()

    llm_with_tools = llm.bind_tools(
        available_tools,
        tool_choice="auto",
    )

    response = llm_with_tools.invoke(
        f"""
        You are an ML monitoring investigation agent.

        Business problem: Predict bank customer churn.

        Detected drift:
        {state["drifted_features"]}

        Completed investigations:
        {state["completed_tools"]}

        Evidence collected:
        {state["investigation_results"]}

        Decide whether another investigation is necessary.

        If additional evidence would improve the investigation,
        call exactly one available tool.

        If the evidence is sufficient, respond with FINISH
        without calling a tool.

        Rules:
        - Never repeat a completed investigation.
        - Never invent model metrics or SHAP findings.
        - Do not confuse customer churn with credit default.
        - Do not retrain or deploy models.
        """
    )

    if response.tool_calls:
        selected = response.tool_calls[0]["name"]

        if selected not in [tool.name for tool in available_tools]:
            raise ValueError(f"Unexpected tool selected: {selected}")

        return {"selected_tool": selected}

    return {"selected_tool": "FINISH"}

def execute_investigation(state: MonitoringState):

    selected = state["selected_tool"]

    if selected in state["completed_tools"]:
        raise ValueError(f"Tool already executed: {selected}")

    if selected == "investigate_feature_importance":

        result = investigate_drift(state)

        evidence = (
            "SHAP investigation completed. "
            f"Top influential features: {result['important_features']}"
        )

        return {
            "important_features": result["important_features"],
            "recommendation": result["recommendation"],
            "tool_result": evidence,
            "completed_tools": state["completed_tools"] + [selected],
            "investigation_results": (
                state["investigation_results"] + [evidence]
            ),
        }

    elif selected == "investigate_distribution_drift":

        comparison = compare_feature_distributions(
            reference_data=state["reference_data"],
            current_data=state["customer_data"],
            drifted_features=state["drifted_features"],
        )

        evidence = f"Distribution comparison: {comparison}"

        return {
            "tool_result": evidence,
            "recommendation": (
                "Review distribution changes and validate "
                "performance on recent labeled data."
            ),
            "completed_tools": state["completed_tools"] + [selected],
            "investigation_results": (
                state["investigation_results"] + [evidence]
            ),
        }

    else:
        raise ValueError(f"Unknown investigation tool: {selected}")

builder = StateGraph(MonitoringState)

builder.add_node("check_alert", check_alert)
builder.add_node("select_investigation_tool", select_investigation_tool)
builder.add_node("execute_investigation", execute_investigation)
builder.add_node("analyze_with_llm", analyze_with_llm)

builder.add_edge(START, "check_alert")


def route_alert(state: MonitoringState):
    if state["status"] == "WARNING":
        return "select_investigation_tool"
    return END


def route_investigation(state: MonitoringState):

    if state["selected_tool"] == "FINISH":
        return "analyze_with_llm"

    if len(state["completed_tools"]) >= 2:
        return "analyze_with_llm"

    return "execute_investigation"


def route_after_execution(state: MonitoringState):

    if len(state["completed_tools"]) >= 2:
        return "analyze_with_llm"

    return "select_investigation_tool"


builder.add_conditional_edges("check_alert", route_alert)

builder.add_conditional_edges(
    "select_investigation_tool",
    route_investigation,
)

builder.add_conditional_edges(
    "execute_investigation",
    route_after_execution,
)

builder.add_edge("analyze_with_llm", END)

graph = builder.compile()