import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Balance relative to salary
    df["BalanceToSalary"] = (
        df["Balance"] / (df["EstimatedSalary"] + 1)
    )

    # Products relative to age
    df["ProductsPerYear"] = (
        df["NumOfProducts"] / df["Age"].replace(0, np.nan)
    )

    # Customer engagement
    df["EngagementScore"] = (
        df["IsActiveMember"] + df["HasCrCard"]
    )

    # Age when customer joined
    df["AgeWhenJoined"] = df["Age"] - df["Tenure"]

    return df