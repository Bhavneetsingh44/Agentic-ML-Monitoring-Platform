from src.models.predict import load_churn_model

model = load_churn_model()

print("Model loaded successfully!")
print(type(model))

print("\nModel input features:")

for feature in model.feature_names_in_:
    print(feature)