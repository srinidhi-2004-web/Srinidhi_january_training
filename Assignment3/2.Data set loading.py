# ---------------------------------------------
# Assignment 03: Linear Regression
# Step 1: Dataset Loading and Target Identification
# File Name: 01_dataset_loading.py
# Author: Srinidhi
# ---------------------------------------------

import pandas as pd

# ---------------------------------------------
# Step 1: Load Dataset
# Dataset: House Price Prediction Dataset (Kaggle)
# Link: https://www.kaggle.com/datasets/shree1992/housedata
# ---------------------------------------------

df = pd.read_csv("data/house_data.csv")

# ---------------------------------------------
# Step 2: Understand Dataset
# ---------------------------------------------

print("✅ Dataset Loaded Successfully!")
print("\n📌 Dataset Shape (Rows, Columns):", df.shape)

print("\n📌 First 5 Rows of Dataset:")
print(df.head())

print("\n📌 Dataset Information:")
print(df.info())

print("\n📌 Statistical Summary:")
print(df.describe())

# ---------------------------------------------
# Step 3: Identify Target Variable
# ---------------------------------------------

target_variable = "price"

print("\n🎯 Target (Output) Variable Identified:")
print("Target Variable =", target_variable)

print("\n📌 Input Features are:")
print(df.drop(target_variable, axis=1).columns)
