# Agentic ML Monitoring & Decision Intelligence Platform

An end-to-end **bank customer churn prediction and monitoring prototype** combining machine learning, experiment tracking, explainability, data-drift detection, LLM-assisted investigation, and API access.

> **Status:** Working locally. Docker, AWS deployment, CI/CD, and production hardening are still in progress. Monitoring examples currently use **deliberately simulated drift**, not live bank traffic.

## What works today

- **Churn model:** scikit-learn Gradient Boosting with feature engineering, categorical encoding and numeric scaling.
- **Experiment tracking:** MLflow for experiments and loading the trained pipeline.
- **Prediction service:** FastAPI `POST /predict` with model explanations; `GET /monitor/drift` for a monitoring demonstration.
- **Data drift:** Evidently detects changes between reference and current feature distributions.
- **Numeric investigation:** Pandas compares mean feature values and relative changes.
- **Explainability:** SHAP ranks influential model features across a sampled set of customers.
- **Agentic workflow:** LangGraph routes alerts, Groq LLM selects up to **two distinct, read-only investigation tools**, and a final LLM step combines the collected evidence.
- **Investigation API:** `POST /agent/investigate` returns status, drifting features, completed tools, SHAP features, investigation evidence, and a narrative report.

## Model results

| Metric | Validation result |
|---|---:|
| ROC-AUC | 0.8502 |
| Accuracy | 0.8485 |
| Precision | 0.6313 |
| Recall | 0.6143 |
| F1 | 0.6227 |
| Decision threshold | 0.30 |

The reported precision, recall, accuracy and F1 correspond to the selected 0.30 threshold. Lowering the threshold improved recall relative to 0.50; metrics should be interpreted with the evaluation split and threshold.

## Architecture

```text
Bank Customer Data
       |
       v
Feature Engineering + sklearn Pipeline
       |
       v
Gradient Boosting + MLflow
       |
       +-------------------> FastAPI /predict
       |
Reference vs Current Data
       |
       v
Evidently Drift Detection
       |
       v
LangGraph Alert Routing
       |
       v
Groq LLM Tool Selection
       |
       +--> Pandas Distribution Comparison
       |
       +--> SHAP Feature Importance
       |
       v
Groq LLM Evidence Summary
       |
       v
FastAPI POST /agent/investigate
```

The LLM can choose to stop after one investigation or request a second distinct investigation. The workflow caps investigation tool executions at two. It **does not automatically retrain or deploy models**.

## Verified simulated-drift example

In the local demonstration, `Age` is increased by 15 and `Balance` is multiplied by 1.5 in the current-data sample. An observed successful API response reported:

| Drifted feature | Reference mean | Current mean | Relative change |
|---|---:|---:|---:|
| Age | 38.94 | 53.86 | +38.32% |
| Balance | 76,800.04 | 112,843.95 | +46.93% |
| ProductsPerYear | ~0.04 | ~0.03 | -29.71% |
| AgeWhenJoined | 33.93 | 48.84 | +43.98% |

The agent completed both `investigate_distribution_drift` and `investigate_feature_importance` in a single run. Its top SHAP-ranked features were `ProductsPerYear`, `Age`, `IsActiveMember`, `Balance`, and `NumOfProducts` (with preprocessing prefixes in raw output).

**Caution:** Drift alone does not establish reduced accuracy or unfair outcomes. Mean changes are diagnostic summaries, not full distribution tests. LLM narrative claims should be checked against the numeric evidence.

## Project layout

```text
src/
├── agents/
│   ├── agent_service.py
│   ├── investigation_tools.py
│   ├── llm_reasoner.py
│   └── monitoring_agent.py
├── api/
│   └── app.py
├── data/
│   ├── feature_engineering.py
│   └── preprocessing.py
├── explainability/
│   └── shap_explainer.py
├── models/
│   ├── predict.py
│   └── train.py
├── monitoring/
│   └── drift.py
└── tracking/
    └── experiment.py
```

Additional local scripts include `test_agent.py`, `test_drift.py`, `test_shap.py`, `test_model.py`, and `main.py`.

## Technology stack

| Area | Tools |
|---|---|
| ML and preprocessing | Python, Pandas, scikit-learn |
| Tracking | MLflow |
| Explainability | SHAP |
| Drift detection | Evidently |
| Agent orchestration | LangGraph, LangChain tools |
| LLM | Groq API (tested with `openai/gpt-oss-20b`) |
| Serving | FastAPI, Pydantic, Uvicorn |
| Version control | Git, GitHub |

## Running locally

1. Install the project's Python dependencies in a virtual environment.
2. Set `GROQ_API_KEY` in a local `.env` file. **Never commit this file.**
3. Ensure the expected dataset and MLflow model artifacts are available locally; model loading currently depends on local MLflow storage.
4. From the repository root, start the API:

```bash
python -m uvicorn src.api.app:app --reload
```

5. Open `http://127.0.0.1:8000/docs` and execute `POST /agent/investigate` to run the **simulated drift** demonstration.

This is not yet a one-command fresh-clone setup: dependency pinning, reproducible model packaging, and deployment configuration remain to be completed.

## Roadmap

- [x] Feature engineering and leakage-aware preprocessing
- [x] Gradient Boosting churn prediction and threshold evaluation
- [x] MLflow experiment tracking and model loading
- [x] FastAPI prediction endpoint
- [x] SHAP explanations
- [x] Evidently drift detection and feature mean comparisons
- [x] Groq integration and LangGraph conditional routing
- [x] Bounded multi-tool agent investigation
- [x] FastAPI `POST /agent/investigate` tested successfully
- [ ] Production-quality drift evaluation, tests, logging and response schemas
- [ ] Docker containerization and reproducible model artifacts
- [ ] AWS deployment
- [ ] CI/CD with GitHub Actions

## Author

**Bhavneet Singh**  
Data Science | Machine Learning | AI Engineering
