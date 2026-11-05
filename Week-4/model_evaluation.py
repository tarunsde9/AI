import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identify feature types
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

# Create baseline model
baseline_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression())
])

print("Project setup completed!")

# Train baseline model
baseline_model.fit(X_train, y_train)

# Predictions
y_pred = baseline_model.predict(X_val)

# Prediction probabilities for positive class
y_prob = baseline_model.predict_proba(X_val)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, pos_label="Yes"))
print("Recall:", recall_score(y_val, y_pred, pos_label="Yes"))
print("F1 Score:", f1_score(y_val, y_pred, pos_label="Yes"))
print("ROC-AUC:", roc_auc_score(
    (y_val == "Yes").astype(int), y_prob
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# Create custom F1 scorer
f1_scorer = make_scorer(
    f1_score,
    pos_label="Yes"
)

# Create cross-validation strategy
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Perform cross-validation
cv_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring=f1_scorer
)

print("Cross-validation scores:", cv_scores)
print("Average F1 Score:", cv_scores.mean())

# Define hyperparameter combinations
parameters = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=parameters,
    cv=cv,
    scoring=f1_scorer,
    n_jobs=-1
)

# Train and tune model
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1:", grid_search.best_score_)

# Get best model
best_model = grid_search.best_estimator_

# Predict validation data
tuned_predictions = best_model.predict(X_val)

# Evaluate tuned model
print("\nTuned Model Evaluation")

print("Accuracy:", accuracy_score(y_val, tuned_predictions))
print("Precision:", precision_score(y_val, tuned_predictions, pos_label="Yes"))
print("Recall:", recall_score(y_val, tuned_predictions, pos_label="Yes"))
print("F1 Score:", f1_score(y_val, tuned_predictions, pos_label="Yes"))

# Create Decision Tree pipeline
decision_tree = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Define Decision Tree parameters
tree_parameters = {
    "classifier__max_depth": [2, 3, 5, None],
    "classifier__min_samples_split": [2, 3, 4]
}

# Tune Decision Tree
tree_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=tree_parameters,
    cv=cv,
    scoring=f1_scorer,
    n_jobs=-1
)

tree_search.fit(X_train, y_train)

# Evaluate Decision Tree
best_tree = tree_search.best_estimator_

tree_predictions = best_tree.predict(X_val)

print("\nDecision Tree Results")
print("Best Parameters:", tree_search.best_params_)
print("Cross-validation F1:", tree_search.best_score_)
print("Validation Accuracy:", accuracy_score(y_val, tree_predictions))
print("Validation F1:", f1_score(y_val, tree_predictions, pos_label="Yes"))