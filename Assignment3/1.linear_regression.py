# Assignment 03: Linear Regression
# Author: Srinidhi
# Dataset: House Price Prediction

# Step 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------

df = pd.read_csv("data/house_data.csv")

print("Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

# --------------------------------------------
# Step 2: Data Cleaning
# --------------------------------------------

print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Fill missing values (if present)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nDataset cleaned successfully!")
print("Updated Shape:", df.shape)

# --------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------------------------

# Target variable
target = "price"

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Relationship between sqft_living and price
plt.figure(figsize=(8,5))
sns.scatterplot(x="sqft_living", y="price", data=df)
plt.title("Price vs Sqft Living")
plt.show()

# --------------------------------------------
# Step 4: Feature Selection
# --------------------------------------------

# Input Features (X) and Target (y)
X = df.drop(target, axis=1)
y = df[target]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# --------------------------------------------
# Step 5: Train-Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# --------------------------------------------
# Step 6: Linear Regression Model
# --------------------------------------------

model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Predictions
y_pred = model.predict(X_test)

# --------------------------------------------
# Step 7: Model Evaluation
# --------------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# --------------------------------------------
# Step 8: Interpretation of Coefficients
# --------------------------------------------

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")
print(coeff_df.sort_values(by="Coefficient", ascending=False))

# --------------------------------------------
# Step 9: Conclusion
# --------------------------------------------

print("\nConclusion:")
print("Lower MSE indicates better prediction accuracy.")
print("R² closer to 1 indicates strong model performance.")
