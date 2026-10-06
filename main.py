import pandas as pd

from src.data.feature_engineering import create_features
from src.data.preprocessing import prepare_data, split_data
from src.tracking.experiment import log_experiment

from src.models.train import (
    build_model,
    train_model,
    evaluate_model,
    compare_thresholds,
    evaluate_at_threshold
)


# STEP 1: Load data
df = pd.read_csv("data/raw/European_Bank.csv")

# STEP 2: Feature engineering
df = create_features(df)

# STEP 3: Separate features and target
X, y = prepare_data(df)

# STEP 4: Train / Validation / Test split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# STEP 5: Build pipeline
model = build_model(X_train)

# STEP 6: Train model
model = train_model(model, X_train, y_train)

# STEP 7: Evaluate at default threshold (0.5)
val_metrics = evaluate_model(model, X_val, y_val)

print("\n--- VALIDATION PERFORMANCE (0.5) ---")

for name, value in val_metrics.items():
    print(f"{name}: {value:.4f}")

# STEP 8: Compare thresholds
print("\n--- THRESHOLD COMPARISON ---")

best_threshold = compare_thresholds(model, X_val, y_val)

print(f"\nSelected threshold: {best_threshold}")

# STEP 9: Evaluate at selected threshold (0.3)
selected_metrics = evaluate_at_threshold(
    model,
    X_val,
    y_val,
    best_threshold
)

print("\n--- SELECTED THRESHOLD METRICS ---")

for name, value in selected_metrics.items():
    print(f"{name}: {value:.4f}")

# STEP 10: Log experiment to MLflow
log_experiment(
    model=model,
    metrics=selected_metrics,
    threshold=best_threshold
)