import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("user_behavior_dataset.csv")


# Encode categorical columns
label_encoders = {}

categorical_columns = [
    "Device Model",
    "Operating System",
    "Gender"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder


# Separate features and target
X = df.drop(columns=["User ID", "User Behavior Class"])
y = df["User Behavior Class"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# Save model
joblib.dump(model, "user_behavior_model.pkl")

# Save encoders
joblib.dump(label_encoders, "label_encoders.pkl")


# Save metrics
metrics = {
    "model": "Random Forest",
    "test_size": 0.2,
    "random_state": 42,
    "accuracy": accuracy
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("Model training completed successfully.")
print("Accuracy:", accuracy)
print("Model saved as user_behavior_model.pkl")
print("Metrics saved as metrics.json")
