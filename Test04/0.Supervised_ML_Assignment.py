"""
Test04 - Supervised Machine Learning Assignment

Objective:
This program demonstrates the implementation of five supervised machine learning algorithms
on a dataset for prediction (classification).

Algorithms Used:
1. Linear Regression (Logistic Regression for classification)
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Dataset Used:
Iris Dataset (from sklearn)

Author: Srinidhi
"""

# -------------------------------
# Import Required Libraries
# -------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
print("Loading Iris Dataset...\n")

iris = load_iris()
X = iris.data      # Features
y = iris.target    # Target labels

# -------------------------------
# Step 2: Split Data into Train and Test
# -------------------------------
print("Splitting dataset into training and testing sets...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 3: Feature Scaling
# -------------------------------
# Scaling is important for KNN and SVM models
print("Applying Feature Scaling...\n")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 4: Define the Models
# -------------------------------
models = {
    "1. Logistic Regression (Linear Model)": LogisticRegression(),
    "2. Decision Tree Classifier": DecisionTreeClassifier(),
    "3. Random Forest Classifier": RandomForestClassifier(),
    "4. K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "5. Support Vector Machine (SVM)": SVC()
}

# -------------------------------
# Step 5: Train and Evaluate Models
# -------------------------------
print("Training and Evaluating Models...\n")

for name, model in models.items():
    print("======================================")
    print(f"Model: {name}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy Score: {accuracy:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

print("======================================")
print("All models trained and evaluated successfully!")
