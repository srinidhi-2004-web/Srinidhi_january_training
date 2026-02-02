# -------------------------------------------------
# Assignment 03: Linear Regression
# Step 2: Data Cleaning
# File Name: 02_data_cleaning.py
# Author: Srinidhi
# -------------------------------------------------

import pandas as pd

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

df = pd.read_csv("data/house_data.csv")

print("✅ Dataset Loaded Successfully!")
print("Original Dataset Shape:", df.shape)

# -------------------------------------------------
# Step 2: Check Missing Values
# -------------------------------------------------

print("\n📌 Missing Values in Dataset:")
print(df.isnull().sum())

# -------------------------------------------------
# Step 3: Remove Duplicate Rows
# -------------------------------------------------

duplicates = df.duplicated().sum()
print("\n📌 Duplicate Rows Found:", duplicates)

# Drop duplicates
df.drop_duplicates(inplace=True)

print("✅ Duplicates Removed!")
print("Dataset Shape After Removing Duplicates:", df.shape)

# -------------------------------------------------
# Step 4: Handle Missing Values
# Strategy: Fill numeric missing values with mean
# -------------------------------------------------

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n✅ Missing Values Handled Successfully!")

# Verify missing values again
print("\n📌 Missing Values After Cleaning:")
print(df.isnull().sum())

# -------------------------------------------------
# Step 5: Save Cleaned Dataset
# -------------------------------------------------

df.to_csv("data/cleaned_house_data.csv", index=False)

print("\n✅ Cleaned Dataset Saved Successfully!")
print("Saved File: data/cleaned_house_data.csv")

# -------------------------------------------------
# Final Dataset Info
# -------------------------------------------------

print("\n📌 Cleaned Dataset Shape:", df.shape)
print("\n📌 First 5 Rows of Cleaned Dataset:")
print(df.head())
