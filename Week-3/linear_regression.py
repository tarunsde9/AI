from sklearn.linear_model import LinearRegression

import numpy as np

# Dataset
X = np.array([1000, 1200, 1500, 1800, 2000])
y = np.array([30, 35, 45, 55, 60])

# Calculate mean values
x_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate weight (slope)
w = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)

# Calculate bias (intercept)
b = y_mean - w * x_mean

print("Weight:", w)
print("Bias:", b)

# Predict house prices
predictions = w * X + b

print("Actual Prices:", y)
print("Predicted Prices:", predictions)

# Calculate Mean Squared Error
mse = np.mean((y - predictions) ** 2)

print("Mean Squared Error:", mse)

# Train using scikit-learn
X_reshaped = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X_reshaped, y)

# Predict prices
sklearn_predictions = model.predict(X_reshaped)

# Calculate MSE
sklearn_mse = np.mean((y - sklearn_predictions) ** 2)

print("\n--- scikit-learn Results ---")
print("Weight:", model.coef_[0])
print("Bias:", model.intercept_)
print("Predicted Prices:", sklearn_predictions)
print("Mean Squared Error:", sklearn_mse)