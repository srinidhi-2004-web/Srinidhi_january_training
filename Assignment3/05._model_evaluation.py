# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 6: Model Evaluation (MSE and R² Score)
# File Name: 06_model_evaluation.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Step 1: Load Actual and Predicted Values
# -------------------------------------------------

predictions = pd.read_csv("data/predictions.csv")

print("✅ Predictions File Loaded Successfully!")
print("\n📌 First 5 Rows:")
print(predictions.head())

# Separate Actual and Predicted values
y_actual = predictions["Actual Price"]
y_predicted = predictions["Predicted Price"]

# -------------------------------------------------
# Step 2: Calculate Mean Squared Error (MSE)
# -------------------------------------------------

mse = mean_squared_error(y_actual, y_predicted)

# -------------------------------------------------
# Step 3: Calculate R² Score
# -------------------------------------------------

r2 = r2_score(y_actual, y_predicted)

# -------------------------------------------------
# Step 4: Display Evaluation Results
# -------------------------------------------------

print("\n✅ Model Evaluation Results:")
print("------------------------------------")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print("------------------------------------")

# -------------------------------------------------
# Step 5: Interpretation of Results
# -------------------------------------------------

print("\n📌 Interpretation:")

if mse > 0:
    print("- Lower MSE indicates the model predictions are closer to actual values.")

if r2 > 0.8:
    print("- R² Score is close to 1, so the model fits the data very well.")
elif r2 > 0.5:
    print("- R² Score is moderate, model performance is acceptable.")
else:
    print("- R² Score is low, model needs improvement.")

print("\n✅ Model Evaluation Completed Successfully!")
