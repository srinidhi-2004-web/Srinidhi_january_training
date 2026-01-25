"""
File Name: additional_steps.py

Purpose:
This file demonstrates Additional Preprocessing Steps:

1. Train/Test Split (Optional)
2. Skewness Transformation using:
   - Log Transformation
   - Power Transformation

These steps help improve ML model performance.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


# ---------------------------------------------------
# 1. Train/Test Split
# ---------------------------------------------------
def train_test_split_data(df, target_column, test_size=0.2):
    """
    Splits dataset into training and testing sets.

    Parameters:
    - df: DataFrame
    - target_column: Column to predict
    - test_size: Percentage of test data

    Returns:
    - X_train, X_test, y_train, y_test
    """
    print("\n--- Train/Test Split ---")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# 2. Log Transformation (Fix Right-Skewed Data)
# ---------------------------------------------------
def log_transformation(df, column):
    """
    Applies log transformation to reduce skewness.

    Formula: log(1 + x)

    Best for positively skewed features.
    """
    print("\n--- Log Transformation ---")

    df[column] = np.log1p(df[column])
    return df


# ---------------------------------------------------
# 3. Power Transformation (Yeo-Johnson)
# ---------------------------------------------------
def power_transformation(df, columns):
    """
    Applies Power Transformation to make data more Gaussian.

    Uses Yeo-Johnson method (works for zero/negative values too).
    """
    print("\n--- Power Transformation ---")

    pt = PowerTransformer(method="yeo-johnson")
    df[columns] = pt.fit_transform(df[columns])

    return df


# ---------------------------------------------------
# Example Usage (Testing)
# ---------------------------------------------------
if __name__ == "__main__":

    # Sample dataset
    data = {
        "Age": [18, 25, 40, 60, 35],
        "Salary": [20000, 30000, 50000, 80000, 45000],
        "LoanAmount": [1000, 5000, 20000, 80000, 15000],
        "Purchased": [1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    print("\nOriginal Dataset:\n", df)

    # Log Transformation for skewed feature
    df_log = log_transformation(df.copy(), "LoanAmount")
    print("\nAfter Log Transformation:\n", df_log)

    # Power Transformation
    df_power = power_transformation(df.copy(), ["Salary", "LoanAmount"])
    print("\nAfter Power Transformation:\n", df_power)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split_data(df, "Purchased")
