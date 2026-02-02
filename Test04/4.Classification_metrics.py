"""
classification_metrics.py

Test04 - Supervised Machine Learning Assignment

This file demonstrates how to evaluate classification models
using mandatory classification metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-Score

Author: Srinidhi
"""

# -------------------------------
# Import Required Libraries
# -------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Sample Actual vs Predicted Labels
# -------------------------------
# These values represent classification output example

y_test = [0, 1, 1, 0, 2, 2, 1, 0]
y_pred = [0, 1, 0, 0, 2, 1, 1, 0]

# -------------------------------
# Classification Metrics Calculation
# -------------------------------

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 2. Precision (Macro Average for multiclass)
precision = precision_score(y_test, y_pred, average="macro")

# 3. Recall (Macro Average for multiclass)
recall = recall_score(y_test, y_pred, average="macro")

# 4. F1 Score (Macro Average for multiclass)
f1 = f1_score(y_test, y_pred, average="macro")

# -------------------------------
# Print Results
# -------------------------------

print("======================================")
print("Classification Evaluation Metrics")
print("======================================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)

print("======================================")
print("Classification Metrics Completed Successfully!")
