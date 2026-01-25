"""
File Name: categorical_encoding.py

Purpose:
This file demonstrates all major categorical encoding techniques:

1. One-Hot Encoding
2. Label Encoding
3. Ordinal Encoding
4. Frequency Encoding
5. Target Encoding

As per project requirement, all encoding methods are implemented separately here.
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ---------------------------------------------------
# 1. One-Hot Encoding
# ---------------------------------------------------
def one_hot_encoding(df, column):
    """
    Converts categorical column into multiple binary columns.
    Best for nominal data (no order).
    """
    print("\n--- One-Hot Encoding ---")
    encoded_df = pd.get_dummies(df, columns=[column])
    return encoded_df


# ---------------------------------------------------
# 2. Label Encoding
# ---------------------------------------------------
def label_encoding(df, column):
    """
    Converts categories into numeric labels (0,1,2,...).
    Suitable for binary categories or ordered categories.
    """
    print("\n--- Label Encoding ---")
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df


# ---------------------------------------------------
# 3. Ordinal Encoding
# ---------------------------------------------------
def ordinal_encoding(df, column, categories):
    """
    Assigns ordered numeric values based on category ranking.
    Example: low < medium < high
    """
    print("\n--- Ordinal Encoding ---")
    oe = OrdinalEncoder(categories=[categories])
    df[column] = oe.fit_transform(df[[column]])
    return df


# ---------------------------------------------------
# 4. Frequency Encoding
# ---------------------------------------------------
def frequency_encoding(df, column):
    """
    Replaces each category with its frequency count.
    Useful when categories repeat often.
    """
    print("\n--- Frequency Encoding ---")
    freq_map = df[column].value_counts().to_dict()
    df[column] = df[column].map(freq_map)
    return df


# ---------------------------------------------------
# 5. Target Encoding
# ---------------------------------------------------
def target_encoding(df, column, target):
    """
    Replaces categories with mean of target variable.
    Useful for high-cardinality categorical features.
    """
    print("\n--- Target Encoding ---")
    target_mean_map = df.groupby(column)[target].mean().to_dict()
    df[column] = df[column].map(target_mean_map)
    return df


# ---------------------------------------------------
# Example Usage (Testing)
# ---------------------------------------------------
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "City": ["Bangalore", "Delhi", "Mumbai", "Delhi", "Mumbai"],
        "Size": ["Small", "Medium", "Large", "Medium", "Small"],
        "Purchased": [1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    print("\nOriginal Dataset:\n", df)

    # One-Hot Encoding
    df_onehot = one_hot_encoding(df.copy(), "City")
    print("\nOne-Hot Encoded Data:\n", df_onehot)

    # Label Encoding
    df_label = label_encoding(df.copy(), "City")
    print("\nLabel Encoded Data:\n", df_label)

    # Ordinal Encoding
    size_order = ["Small", "Medium", "Large"]
    df_ordinal = ordinal_encoding(df.copy(), "Size", size_order)
    print("\nOrdinal Encoded Data:\n", df_ordinal)

    # Frequency Encoding
    df_freq = frequency_encoding(df.copy(), "City")
    print("\nFrequency Encoded Data:\n", df_freq)

    # Target Encoding
    df_target = target_encoding(df.copy(), "City", "Purchased")
    print("\nTarget Encoded Data:\n", df_target)
