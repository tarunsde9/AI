import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2

# Load dataset
df = pd.read_csv("dataset.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

print("\nTraining target shape:", y_train.shape)
print("Validation target shape:", y_val.shape)

# Build neural network
model = Sequential([
    Dense(16, activation="relu", input_shape=(5,)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Configure model training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.show()


# Build model with Dropout
dropout_model = Sequential([
    Input(shape=(5,)),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

dropout_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Dropout model
dropout_history = dropout_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Build model with L2 regularization
l2_model = Sequential([
    Input(shape=(5,)),
    Dense(16, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(8, activation="relu", kernel_regularizer=l2(0.001)),
    Dense(1, activation="sigmoid")
])

l2_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train L2 model
l2_history = l2_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Compare all three models
models = {
    "Baseline": history,
    "Dropout": dropout_history,
    "L2": l2_history
}

print("\n========== MODEL COMPARISON ==========")

for name, result in models.items():
    print(f"\n{name} Model")
    print(f"Training Loss: {result.history['loss'][-1]:.4f}")
    print(f"Validation Loss: {result.history['val_loss'][-1]:.4f}")
    print(f"Training Accuracy: {result.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {result.history['val_accuracy'][-1]:.4f}")


plt.figure(figsize=(9, 5))

for name, result in models.items():
    plt.plot(result.history["val_loss"], label=name)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Use the L2 model for final evaluation
predictions = l2_model.predict(X_val)

predicted_classes = (predictions >= 0.5).astype(int)

print("\n========== L2 MODEL EVALUATION ==========")
print(classification_report(y_val, predicted_classes))
