# Import necessary libraries
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Convert the data into a DataFrame for better visualization and manipulation
columns = [f'feature_{i}' for i in range(diabetes.data.shape[1])]
data = pd.DataFrame(data=diabetes.data, columns=columns)
data['target'] = diabetes.target

# Convert target to binary classes (e.g., above/below the median)
data['target_binary'] = (
    data['target'] > data['target'].median()
).astype(int)

# Split the data into training and testing sets
X = data.drop(columns=['target', 'target_binary'])  # Features
y = data['target_binary']  # Binary target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate and print the performance metrics
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plotting the results
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Target Classes')
plt.ylabel('Predicted Target Classes')
plt.title('Logistic Regression: Predicted vs Actual')
plt.show()
