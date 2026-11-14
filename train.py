
# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import os

# 2. Load the dataset
iris = load_iris()

X = iris.data
y = iris.target

# 3. Convert into DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Initialize the ML model
model = KNeighborsClassifier(n_neighbors=3)

# 6. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/iris_model.pkl")

print("\nModel saved successfully!")
