# Agentic ML Monitoring & Decision Intelligence Platform

An end-to-end machine learning platform for predicting bank customer churn and evolving the model from a notebook-style experiment into a production-oriented ML system with experiment tracking, API-based inference, explainability, monitoring, and an agentic decision layer.

## Project Goal

The project predicts customers at risk of churn and turns model output into a usable decision-intelligence workflow. It is designed to demonstrate the full ML lifecycle rather than stopping at model training.

## Current ML Results

| Metric | Validation Result |
|---|---:|
| ROC-AUC | 0.8502 |
| Accuracy | 0.8485 |
| Precision | 0.6313 |
| Recall | 0.6143 |
| F1 Score | 0.6227 |
| Decision Threshold | 0.30 |

The 0.30 threshold was selected from tested thresholds (0.30, 0.40 and 0.50) using validation F1. Lowering the threshold improved churn recall compared with the default 0.50 threshold.

## Current Architecture

```text
Raw Customer Data
       |
       v
Feature Engineering
       |
       v
Train / Validation / Test Split
       |
       v
ColumnTransformer
  |              |
OneHotEncoder  StandardScaler
       |
       v
Gradient Boosting Classifier
       |
       v
MLflow Experiment Tracking
       |
       v
FastAPI Prediction API
       |
       v
Decision Intelligence / Monitoring (in progress)
```

## Features Engineered

- `BalanceToSalary`
- `ProductsPerYear`
- `EngagementScore`
- `AgeWhenJoined`

Categorical variables (`Geography`, `Gender`) are encoded with `OneHotEncoder`, while numerical variables are standardized with `StandardScaler` inside a scikit-learn `ColumnTransformer` and `Pipeline`.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data | Pandas |
| Machine Learning | scikit-learn, Gradient Boosting |
| Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Version Control | Git, GitHub |
| Planned | SHAP, monitoring, agentic AI, Docker, AWS, CI/CD |

## Project Structure

```text
Agentic-ML-Monitoring-Platform/
├── data/
│   └── raw/
│       └── European_Bank.csv
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── tracking/
│       └── experiment.py
├── main.py
├── test_model.py
├── .gitignore
└── README.md
```

## ML Workflow

1. Load bank customer data.
2. Engineer churn-related behavioral features.
3. Split the data into training, validation and test sets using stratification.
4. Build preprocessing and model steps as a scikit-learn pipeline to reduce data-leakage risk.
5. Train a Gradient Boosting classifier.
6. Evaluate probabilities using ROC-AUC, precision, recall and F1.
7. Compare decision thresholds on the validation set and select 0.30 based on validation F1 among the tested thresholds.
8. Track experiments and the trained pipeline with MLflow.
9. Load the tracked model for inference.
10. Expose churn predictions through FastAPI.

## API

The FastAPI service accepts customer information and returns the predicted churn probability and threshold-based churn decision.

Run locally with:

```bash
python -m uvicorn src.api.app:app --reload
```

Then open the interactive API documentation at `/docs` on the local server.

## Development Roadmap

- [x] Feature engineering
- [x] Train / validation / test workflow
- [x] Scikit-learn preprocessing pipeline
- [x] Gradient Boosting churn model
- [x] Threshold evaluation
- [x] MLflow experiment tracking
- [x] FastAPI inference layer
- [ ] SHAP explainability
- [ ] Model/data drift monitoring
- [ ] Agentic incident analysis and decision support
- [ ] Docker containerization
- [ ] AWS deployment
- [ ] CI/CD with GitHub Actions

## Why This Project

A model is only one part of a real ML system. This project is being built to demonstrate how a prediction model can be tracked, served, explained, monitored and eventually connected to an agentic decision workflow.

## Author

**Bhavneet Singh**  
Data Science | Machine Learning | AI Engineering
