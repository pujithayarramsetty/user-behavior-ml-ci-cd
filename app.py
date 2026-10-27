
import os
import time
import uuid
import logging

import joblib
import pandas as pd

from flask import Flask, request, jsonify, g
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load trained model and label encoders
# --------------------------------------------------

MODEL_PATH = "user_behavior_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# Get the exact feature names and order used during training.
FEATURE_NAMES = list(model.feature_names_in_)

logger.info("Model and label encoders loaded successfully")
logger.info("Expected features: %s", FEATURE_NAMES)

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
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())


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
            g, "start_time", time.time()
        )

        REQUEST_LATENCY.observe(duration)

    response.headers["X-Request-ID"] = getattr(
        g, "request_id", "unknown"
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
                "error": "Request body must be a valid JSON object"
            }), 400

        # Check for missing model features.
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

        # Process features in the exact training order.
        processed_data = {}

        for column in FEATURE_NAMES:
            value = data[column]

            # Encode categorical features using saved encoders.
            if column in label_encoders:
                value = str(value)
                encoder = label_encoders[column]

                known_classes = [
                    str(item) for item in encoder.classes_
                ]

                if value not in known_classes:
                    return jsonify({
                        "error": f"Unknown value for {column}",
                        "received_value": value,
                        "allowed_values": known_classes
                    }), 400

                value = encoder.transform([value])[0]

            processed_data[column] = value

        # Explicitly preserve the model's feature names and order.
        input_df = pd.DataFrame(
            [processed_data],
            columns=FEATURE_NAMES
        )

        # Generate prediction.
        prediction = model.predict(input_df)[0]

        # Convert NumPy scalar to a standard Python value.
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        logger.info(
            "request_id=%s prediction completed successfully",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "prediction": prediction,
            "message": "Prediction successful",
            "request_id": g.request_id
        }), 200

    except Exception:
        logger.exception(
            "request_id=%s prediction failed",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "error": "Prediction failed. Check application logs."
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
