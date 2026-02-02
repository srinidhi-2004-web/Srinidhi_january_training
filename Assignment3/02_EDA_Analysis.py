# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 3: Exploratory Data Analysis (EDA)
# File Name: 03_eda_analysis.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Step 1: Load Cleaned Dataset
# -------------------------------------------------

df = pd.read_csv("data/cleaned_house_data.csv")

print("✅ Cleaned Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Target Variable
target = "price"

# -------------------------------------------------
# Step 2: Basic Dataset Overview
# -------------------------------------------------

print("\n📌 First 5 Rows:")
print(df.head())

print("\n📌 Dataset Summary:")
print(df.describe())

# -------------------------------------------------
# Step 3: Correlation Analysis (Multicollinearity Check)
# -------------------------------------------------

print("\n📌 Correlation Matrix:")

correlation_matrix = df.corr()
print(correlation_matrix)

# Heatmap Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Multicollinearity Check)")
plt.show()

# -------------------------------------------------
# Step 4: Relationship Between Features and Target
# -------------------------------------------------

# Select top important numeric features
features = ["sqft_living", "bedrooms", "bathrooms", "floors"]

for feature in features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f"{target} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()

# -------------------------------------------------
# Step 5: Distribution of Target Variable
# -------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df[target], kde=True)
plt.title("Distribution of House Prices (Target Variable)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# -------------------------------------------------
# Step 6: Pairplot for Feature Relationships
# -------------------------------------------------

print("\n📌 Pairplot to Understand Feature Relationships")

sns.pairplot(df[[target, "sqft_living", "bedrooms", "bathrooms"]])
plt.show()

# -------------------------------------------------
# Conclusion from EDA
# -------------------------------------------------

print("\n✅ EDA Completed Successfully!")
print("Key Observations:")
print("- Correlation heatmap helps detect multicollinearity.")
print("- Scatterplots show relationship between features and target.")
print("- sqft_living is strongly correlated with price.")
