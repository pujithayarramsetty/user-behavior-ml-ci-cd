
import os
import time
import logging
import uuid

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------- Prometheus Metrics ----------------
REQUEST_COUNT = Counter(
    "flask_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "flask_request_duration_seconds",
    "HTTP request processing time"
)

# ---------------- Load ML Model and Encoders ----------------
try:
    model = joblib.load("user_behavior_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    logger.info("Model and label encoders loaded successfully")
except Exception:
    logger.exception("Unable to load model or label encoders")
    raise


# ---------------- Request Monitoring ----------------
@app.before_request
def start_request_timer():
    g.request_start_time = time.perf_counter()
    g.request_id = str(uuid.uuid4())


@app.after_request
def record_request_metrics(response):
    start_time = getattr(g, "request_start_time", None)

    if start_time is not None and request.path != "/metrics":
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            status=str(response.status_code)
        ).inc()

        REQUEST_LATENCY.observe(duration)

        logger.info(
            "request_id=%s method=%s path=%s status=%s duration=%.4fs",
            getattr(g, "request_id", "unknown"),
            request.method,
            request.path,
            response.status_code,
            duration
        )

    return response


# ---------------- Home Endpoint ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "User Behavior ML API is running",
        "status": "ok"
    }), 200


# ---------------- Health Endpoint ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


# ---------------- Prometheus Metrics Endpoint ----------------
@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )


# ---------------- Prediction Endpoint ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or not data:
            return jsonify({
                "error": "Request body must contain a non-empty JSON object"
            }), 400

        required_fields = [
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

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Encode categorical inputs using the saved encoders
        processed_data = {}

        for column, value in data.items():
            if column in label_encoders:
                processed_data[column] = (
                    label_encoders[column].transform([str(value)])[0]
                )
            else:
                processed_data[column] = value

        input_df = pd.DataFrame([processed_data])

        # Match the exact feature order used during training
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)
            missing_model_columns = [
                column for column in expected_columns
                if column not in input_df.columns
            ]

            if missing_model_columns:
                logger.error(
                    "Missing model features: %s",
                    missing_model_columns
                )
                return jsonify({
                    "error": "Input does not contain all required model features",
                    "missing_fields": missing_model_columns
                }), 400

            input_df = input_df[expected_columns]

        prediction = model.predict(input_df)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        logger.info(
            "request_id=%s prediction completed successfully",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "prediction": prediction
        }), 200

    except ValueError as error:
        logger.warning(
            "request_id=%s invalid prediction input: %s",
            getattr(g, "request_id", "unknown"),
            str(error)
        )
        return jsonify({
            "error": "Invalid input value",
            "details": str(error)
        }), 400

    except Exception:
        logger.exception(
            "request_id=%s prediction failed",
            getattr(g, "request_id", "unknown")
        )
        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500


# ---------------- Run Application ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
