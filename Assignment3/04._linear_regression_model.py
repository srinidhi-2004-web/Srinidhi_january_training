# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 5: Linear Regression Model Training & Prediction
# File Name: 05_linear_regression_model.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Step 1: Load Training and Testing Data
# -------------------------------------------------

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

print("✅ Training and Testing Data Loaded Successfully!")

print("\nTraining Data Shape:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data Shape:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# -------------------------------------------------
# Step 2: Build Linear Regression Model
# -------------------------------------------------

model = LinearRegression()

print("\n✅ Linear Regression Model Created!")

# -------------------------------------------------
# Step 3: Train the Model Using Training Data
# -------------------------------------------------

model.fit(X_train, y_train)

print("✅ Model Training Completed Successfully!")

# -------------------------------------------------
# Step 4: Make Predictions on Test Data
# -------------------------------------------------

y_pred = model.predict(X_test)

print("\n✅ Predictions Completed!")

# -------------------------------------------------
# Step 5: Save Predictions for Evaluation
# -------------------------------------------------

predictions = pd.DataFrame({
    "Actual Price": y_test.values.flatten(),
    "Predicted Price": y_pred.flatten()
})

predictions.to_csv("data/predictions.csv", index=False)

print("\n✅ Predictions Saved Successfully!")
print("Saved File: data/predictions.csv")

# Display first 5 predictions
print("\n📌 Sample Predictions:")
print(predictions.head())
