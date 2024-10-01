# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a logistic regression model
logistic_regression = LogisticRegression(
    solver='lbfgs', multi_class='auto', max_iter=10000)

# Fit the model to the training data
logistic_regression.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_regression.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f'Accuracy: {accuracy:.2f}')
print(report)
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.title("Logistic Regression Model")
plt.show()
