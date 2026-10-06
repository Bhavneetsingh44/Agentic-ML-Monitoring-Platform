import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df: pd.DataFrame):
    df = df.copy()

    # Remove columns that should not be used for prediction
    columns_to_drop = ["CustomerId", "Surname"]
    df = df.drop(columns=columns_to_drop)

    # Separate input features and target
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    return X, y


def build_preprocessor(X):

    categorical_features = ["Geography", "Gender"]

    numerical_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    return preprocessor

def split_data(X, y):

    # First split: 80% development, 20% testing
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Second split: 75% training, 25% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=42,
        stratify=y_dev
    )

    return X_train, X_val, X_test, y_train, y_val, y_test