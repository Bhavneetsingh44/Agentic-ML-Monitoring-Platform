import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the bank churn model.
    """

    # Work on a copy so the original dataframe is not modified
    df = df.copy()

    # Balance relative to salary
    df["BalanceToSalary"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Number of products relative to customer age
    df["ProductsPerYear"] = df["NumOfProducts"] / df["Age"]

    # Simple customer engagement indicator
    df["EngagementScore"] = (
        df["IsActiveMember"] +
        df["HasCrCard"]
    )

    # Approximate age when customer joined the bank
    df["AgeWhenJoined"] = df["Age"] - df["Tenure"]

    return df