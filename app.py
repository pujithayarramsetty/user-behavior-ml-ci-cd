from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_PATH = Path("user_behavior_model.pkl")
ENCODER_PATH = Path("label_encoders.pkl")

FEATURES = [
    "Device Model",
    "Operating System",
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
    "Gender"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "user_behavior_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


def load_encoders():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "label_encoders.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(ENCODER_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "user-behavior-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    encoders = load_encoders()

    sample_data = {}

    for feature in FEATURES:

        if feature in encoders:
            try:
                sample_data[feature] = encoders[feature].transform(
                    [data[feature]]
                )[0]
            except ValueError:
                return jsonify({
                    "error": f"Unknown value for {feature}"
                }), 400
        else:
            sample_data[feature] = data[feature]

    sample = pd.DataFrame([sample_data])

    model = load_model()

    prediction = int(model.predict(sample)[0])

    return jsonify({
        "prediction": prediction,
        "prediction_type": "User Behavior Class"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
