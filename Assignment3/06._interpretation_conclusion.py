# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 7: Interpretation & Conclusion
# File Name: 07_interpretation_conclusion.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# Step 1: Load Training Data Again
# -------------------------------------------------

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

print("✅ Training Data Loaded Successfully!")

# -------------------------------------------------
# Step 2: Train Linear Regression Model Again
# -------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Linear Regression Model Trained Successfully!")

# -------------------------------------------------
# Step 3: Extract Feature Coefficients
# -------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

# Sort coefficients (important features first)
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

print("\n📌 Feature Coefficients (Impact on Target Variable):")
print("------------------------------------")
print(coefficients)
print("------------------------------------")

# -------------------------------------------------
# Step 4: Interpretation of Coefficients
# -------------------------------------------------

print("\n📌 Interpretation:")

print("\nPositive Coefficient Features:")
print("These features increase house price when their value increases.\n")

positive_features = coefficients[coefficients["Coefficient"] > 0]
print(positive_features)

print("\nNegative Coefficient Features:")
print("These features decrease house price when their value increases.\n")

negative_features = coefficients[coefficients["Coefficient"] < 0]
print(negative_features)

# -------------------------------------------------
# Step 5: Conclusion Statement
# -------------------------------------------------

print("\n✅ Final Conclusion:")
print("-------------------------------------------------")
print("1. Linear Regression was successfully applied to predict house prices.")
print("2. Coefficients show how each feature affects the target variable (price).")
print("3. Features like sqft_living have strong positive influence on house price.")
print("4. Negative coefficients indicate features that reduce predicted price.")
print("5. The model evaluation metrics (MSE and R²) confirm model performance.")
print("-------------------------------------------------")

print("\n✅ Interpretation & Conclusion Completed Successfully!")
