import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the gene expression dataset
data = pd.read_csv('gene_expression.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Prepare the features and target variable
# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target (last column)

# Split the data into training and testing sets
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
# Since gene expression datasets can be high-dimensional,
# we'll use a subset of features for visualization if needed.
# Here we plot just the first two features for simplicity.
plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test,
            cmap='coolwarm', label='Actual')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=predictions, marker='x', cmap='coolwarm', label='Predicted')
plt.xlabel('Gene Expression 1')
plt.ylabel('Gene Expression 2')
plt.title('Logistic Regression: Actual vs Predicted (Subset of Features)')
plt.legend()
plt.colorbar(label='Condition')
plt.show()
