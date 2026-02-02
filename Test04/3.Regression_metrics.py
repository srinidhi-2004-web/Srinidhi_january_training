"""
regression_metrics.py

Test04 - Supervised Machine Learning Assignment

This file demonstrates how to evaluate regression models
using mandatory regression metrics:

1. R² Score
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. Mean Absolute Error (MAE)

Author: Srinidhi
"""

# -------------------------------
# Import Required Libraries
# -------------------------------

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# -------------------------------
# Sample Actual vs Predicted Values
# -------------------------------
# These values represent regression output example

y_test = [100, 150, 200, 250, 300]
y_pred = [110, 140, 210, 240, 310]

# -------------------------------
# Regression Metrics Calculation
# -------------------------------

# 1. R² Score
r2 = r2_score(y_test, y_pred)

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# 4. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# -------------------------------
# Print Results
# -------------------------------

print("======================================")
print("Regression Evaluation Metrics Results")
print("======================================")

print("R² Score :", r2)
print("MSE      :", mse)
print("RMSE     :", rmse)
print("MAE      :", mae)

print("======================================")
print("Regression Metrics Completed Successfully!")
