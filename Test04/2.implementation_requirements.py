"""
implementation_requirements.py

Test04 - Supervised Machine Learning Assignment

This file satisfies the Implementation Requirements mentioned in the PDF:

Requirements Covered:
1. Use Python libraries: pandas, numpy, matplotlib, scikit-learn
2. Train and test all five supervised algorithms
3. Compare model performance using evaluation metrics

Algorithms Implemented:
1. Logistic Regression (Linear Model)
2. Decision Tree
3. Random Forest
4. KNN
5. SVM

Evaluation Metrics Used:
- Accuracy
- Precision
- Recall
- F1-Score

Author: Srinidhi
"""

# -------------------------------
# Import Required Libraries
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
print("\nLoading Iris Dataset...")

iris = load_iris()
X = iris.data
y = iris.target

# -------------------------------
# Step 2: Train-Test Split
# -------------------------------
print("Splitting dataset into Train and Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 3: Feature Scaling
# -------------------------------
print("Applying Feature Scaling...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 4: Define All 5 Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

# -------------------------------
# Step 5: Train, Test, Evaluate All Models
# -------------------------------
print("\nTraining and Evaluating Models...\n")

results = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Store Results
    results.append([name, acc, prec, rec, f1])

    print("======================================")
    print("Model:", name)
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

# -------------------------------
# Step 6: Compare Model Performance
# -------------------------------
print("\n======================================")
print("Final Model Comparison Table\n")

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print(results_df)

# -------------------------------
# Step 7: Visualization (Optional)
# -------------------------------
print("\nPlotting Accuracy Comparison...\n")

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Accuracy Comparison of ML Models")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=30)
plt.show()

print("\nImplementation Requirements Completed Successfully!")
