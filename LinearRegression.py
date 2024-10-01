# Import necessary libraries
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Convert the data into a DataFrame for better visualization and manipulation
columns = [f'feature_{i}' for i in range(diabetes.data.shape[1])]
data = pd.DataFrame(data=diabetes.data, columns=columns)
data['target'] = diabetes.target

# Split the data into training and testing sets
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Plotting the results
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.show()
