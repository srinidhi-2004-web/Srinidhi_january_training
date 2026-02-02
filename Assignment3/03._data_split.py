# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 4: Data Split (Train-Test Split)
# File Name: 04_data_split.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Step 1: Load Cleaned Dataset
# -------------------------------------------------

df = pd.read_csv("data/cleaned_house_data.csv")

print("✅ Cleaned Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# -------------------------------------------------
# Step 2: Identify Target Variable
# -------------------------------------------------

target = "price"

# Input Features (X) and Output Target (y)
X = df.drop(target, axis=1)
y = df[target]

print("\n📌 Input Features Shape (X):", X.shape)
print("📌 Target Variable Shape (y):", y.shape)

# -------------------------------------------------
# Step 3: Split Dataset into Training and Testing
# Training: 80%
# Testing: 20%
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data Split Completed Successfully!")

print("\n📌 Training Set Size:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n📌 Testing Set Size:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# -------------------------------------------------
# Step 4: Save Split Data for Next Steps
# -------------------------------------------------

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\n✅ Training and Testing Data Saved Successfully!")
print("Saved Files:")
print("- data/X_train.csv")
print("- data/X_test.csv")
print("- data/y_train.csv")
print("- data/y_test.csv")
