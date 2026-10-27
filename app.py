import os
import time
import uuid
import logging

import joblib
import pandas as pd

from flask import Flask, request, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load the trained model and encoders
# --------------------------------------------------

MODEL_PATH = "user_behavior_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# Get the exact feature names and order used during training.
FEATURE_NAMES = list(model.feature_names_in_)

logger.info("Model loaded successfully")
logger.info("Expected model features: %s", FEATURE_NAMES)

# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

REQUEST_COUNT = Counter(
    "flask_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "flask_request_duration_seconds",
    "HTTP request duration in seconds"
)

# --------------------------------------------------
# Request tracking
# --------------------------------------------------

@app.before_request
def before_request():
    request.start_time = time.time()
    request.request_id = str(uuid.uuid4())


@app.after_request
def after_request(response):

    endpoint = request.endpoint or "unknown"

    if request.path != "/metrics":

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code)
        ).inc()

        duration = time.time() - getattr(
            request,
            "start_time",
            time.time()
        )

        REQUEST_LATENCY.observe(duration)

    response.headers["X-Request-ID"] = getattr(
        request,
        "request_id",
        "unknown"
    )

    return response

# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "User Behavior ML API is running"
    })

# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object"
            }), 400

        # Check that every required feature is present.
        missing_features = [
            feature
            for feature in FEATURE_NAMES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features,
                "expected_features": FEATURE_NAMES
            }), 400

        # Arrange the values in the exact training order.
        input_data = {
            feature: data[feature]
            for feature in FEATURE_NAMES
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURE_NAMES
        )

        # Encode categorical inputs using the saved encoders.
        for column, encoder in label_encoders.items():

            if column not in input_df.columns:
                continue

            value = str(input_df.at[0, column])

            known_classes = [
                str(item)
                for item in encoder.classes_
            ]

            if value not in known_classes:
                return jsonify({
                    "error": f"Unknown value for {column}",
                    "received_value": value,
                    "allowed_values": known_classes
                }), 400

            input_df[column] = encoder.transform(
                input_df[column].astype(str)
            )

        # Ensure the input order exactly matches model training.
        input_df = input_df[FEATURE_NAMES]

        # Predict the user behavior class.
        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "message": "Prediction successful",
            "request_id": request.request_id
        })

    except Exception as error:

        logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500

# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------

@app.route("/metrics", methods=["GET"])
def metrics():

    return (
        generate_latest(),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST}
    )

# --------------------------------------------------
# Run the Flask application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
