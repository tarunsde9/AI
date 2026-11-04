import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Display dataset
print(df)

# Display dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Display statistical summary
print(df.describe())

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features:")
print(X)

print("\nTarget:")
print(y)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# Identify numerical and categorical columns
numerical_features = ["Age", "MonthlyCharges", "Tenure"]
categorical_features = ["Contract"]

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("Preprocessing setup completed!")

# Create classification pipeline
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

print("Classification pipeline created successfully!")

# Train model
model_pipeline.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model_pipeline.predict(X_val)

# Evaluate model
accuracy = accuracy_score(y_val, y_pred)

print("Actual Values:", list(y_val))
print("Predicted Values:", list(y_pred))

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))