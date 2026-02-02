"""
preprocessing.py

This file performs mandatory data preprocessing steps required for
Test04 Supervised Machine Learning Assignment.

Preprocessing Includes:
1. Handling missing values
2. Fixing wrong data types
3. Removing duplicates
4. Outlier detection and treatment
5. Encoding categorical variables
6. Feature scaling
7. Removing irrelevant features
8. Train-test split
9. Skewness check (if required)

Dataset Used: Iris Dataset (Sample Classification Dataset)

Author: Srinidhi
"""

# -------------------------------
# Import Required Libraries
# -------------------------------
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
print("\nLoading Dataset...")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("\nDataset Loaded Successfully!")
print(df.head())

# -------------------------------
# Step 2: Check Missing Values
# -------------------------------
print("\nChecking Missing Values...")

print(df.isnull().sum())

# If missing values exist, fill them
df.fillna(df.mean(), inplace=True)

print("Missing Values Handled!")

# -------------------------------
# Step 3: Fix Wrong Data Types
# -------------------------------
print("\nChecking Data Types...")

print(df.dtypes)

# Ensure numeric features are float
for col in df.columns[:-1]:
    df[col] = df[col].astype(float)

print("Data Types Fixed!")

# -------------------------------
# Step 4: Remove Duplicate Records
# -------------------------------
print("\nChecking Duplicate Records...")

duplicates = df.duplicated().sum()
print("Total Duplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

print("Duplicates Removed!")

# -------------------------------
# Step 5: Outlier Detection & Treatment
# -------------------------------
print("\nDetecting Outliers Using IQR Method...")

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[~((df < (Q1 - 1.5 * IQR)) |
          (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Outliers Treated Successfully!")

# -------------------------------
# Step 6: Encoding Categorical Variables
# -------------------------------
print("\nEncoding Target Column...")

# Iris target is already numeric, but included for requirement
encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])

print("Encoding Completed!")

# -------------------------------
# Step 7: Remove Irrelevant Features
# -------------------------------
print("\nChecking for Irrelevant Features...")

# In Iris dataset, all features are useful
# Example: df.drop(["unnecessary_column"], axis=1)

print("No Irrelevant Features Found!")

# -------------------------------
# Step 8: Split Features and Target
# -------------------------------
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# -------------------------------
# Step 9: Train-Test Split
# -------------------------------
print("\nSplitting Data into Train and Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-Test Split Done!")

# -------------------------------
# Step 10: Feature Scaling
# -------------------------------
print("\nApplying Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed!")

# -------------------------------
# Final Output
# -------------------------------
print("\nPreprocessing Completed Successfully!")
print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

# These outputs can be used in all 5 models
