from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.data.preprocessing import build_preprocessor


def build_model(X_train):

    # Create our preprocessing system
    preprocessor = build_preprocessor(X_train)

    # Combine preprocessing + ML model
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ]
    )

    return model_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics

def compare_thresholds(model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = None
    best_f1 = -1

    for threshold in [0.3, 0.4, 0.5]:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        f1 = f1_score(y_val, predictions)

        print(f"\nThreshold: {threshold}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def evaluate_at_threshold(model, X, y, threshold):

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1": f1_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities)
    }

    return metrics