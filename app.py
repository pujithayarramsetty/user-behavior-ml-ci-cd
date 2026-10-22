from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import logging
import os
import time
import uuid

from flask import Flask, request, jsonify, g
import pandas as pd
import joblib

app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "flask_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "flask_request_duration_seconds",
    "HTTP request processing time"
)


# Configure logging for Render and local execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Load trained model and label encoders
try:
    model = joblib.load("user_behavior_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    logger.info("Model and label encoders loaded successfully.")
except Exception:
    logger.exception("Failed to load model or label encoders.")
    raise


# Log every request when it starts
@app.before_request
def log_request_start():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()

    logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


# Log response status and execution time
@app.after_request
def log_request_end(response):
    duration = time.time() - getattr(g, "start_time", time.time())

    logger.info(
        "Request completed | id=%s method=%s path=%s "
        "status=%s duration=%.3fs",
        getattr(g, "request_id", "unknown"),
        request.method,
        request.path,
        response.status_code,
        duration
    )

    response.headers["X-Request-ID"] = getattr(
        g, "request_id", "unknown"
    )

    return response


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    logger.info("Home endpoint accessed.")
    return jsonify({"status": "ok"})


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    logger.info("Health check endpoint accessed.")
    return jsonify({"status": "healthy"})


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        # Validate JSON input
        if not isinstance(data, dict) or not data:
            logger.warning(
                "Prediction request contains invalid or empty JSON."
            )
            return jsonify({
                "error": "No valid JSON object provided"
            }), 400

        # Required input fields
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

        # Check missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            logger.warning(
                "Prediction request missing required fields: %s",
                missing_fields
            )
            return jsonify({
                "error": "Missing required features",
                "missing_fields": missing_fields
            }), 400

        # Prepare input features
        try:
            input_data = pd.DataFrame([{
                "Device Model": data["Device Model"],
                "Operating System": data["Operating System"],
                "App Usage Time (min/day)": float(
                    data["App Usage Time (min/day)"]
                ),
                "Screen On Time (hours/day)": float(
                    data["Screen On Time (hours/day)"]
                ),
                "Battery Drain (mAh/day)": float(
                    data["Battery Drain (mAh/day)"]
                ),
                "Number of Apps Installed": int(
                    data["Number of Apps Installed"]
                ),
                "Data Usage (MB/day)": float(
                    data["Data Usage (MB/day)"]
                ),
                "Age": int(data["Age"]),
                "Gender": data["Gender"]
            }])

        except (ValueError, TypeError, OverflowError):
            logger.warning(
                "Prediction request contains invalid numeric values."
            )
            return jsonify({
                "error": "Invalid numeric input values"
            }), 400

        # Encode categorical features
        categorical_columns = [
            "Device Model",
            "Operating System",
            "Gender"
        ]

        for column in categorical_columns:
            encoder = label_encoders[column]

            try:
                input_data[column] = encoder.transform(
                    input_data[column]
                )

            except ValueError:
                logger.warning(
                    "Unknown categorical value for feature: %s",
                    column
                )
                return jsonify({
                    "error": f"Unknown value for {column}"
                }), 400

        # Generate prediction
        prediction = model.predict(input_data)[0]

        logger.info(
            "Prediction completed successfully | prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception:
        logger.exception(
            "Unexpected prediction error | request_id=%s",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "error": "Internal server error",
            "request_id": getattr(g, "request_id", "unknown")
        }), 500


# Start Flask application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
