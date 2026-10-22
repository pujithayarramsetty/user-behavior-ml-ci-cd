
import os
import time
import logging
import uuid

import joblib
from flask import Flask, request, jsonify, g, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

# Load trained model and encoders
try:
    model = joblib.load("user_behavior_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    logger.info("Model and label encoders loaded successfully")
except Exception:
    logger.exception("Failed to load model or label encoders")
    raise


# Start request timer and assign request ID
@app.before_request
def start_request_timer():
    g.request_start_time = time.perf_counter()
    g.request_id = str(uuid.uuid4())


# Record request metrics and log request duration
@app.after_request
def record_request_metrics(response):
    start_time = getattr(g, "request_start_time", None)

    if start_time is not None and request.path != "/metrics":
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
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


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "User Behavior ML API is running",
        "status": "ok"
    })


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


# Prometheus metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            logger.warning("Prediction request received without valid JSON")
            return jsonify({
                "error": "Request body must contain valid JSON"
            }), 400

        # Convert input categories using saved encoders
        processed_data = {}

        for column, value in data.items():
            if column in label_encoders:
                processed_data[column] = label_encoders[column].transform(
                    [str(value)]
                )[0]
            else:
                processed_data[column] = value

        import pandas as pd

        input_df = pd.DataFrame([processed_data])
        prediction = model.predict(input_df)

        result = prediction[0]

        # Convert NumPy values to normal Python values if needed
        if hasattr(result, "item"):
            result = result.item()

        logger.info(
            "request_id=%s prediction completed successfully",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "prediction": result
        })

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
