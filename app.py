from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained User Behavior model
model = joblib.load("user_behavior_model.pkl")


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400

        # Required input features
        required_features = [
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

        # Check for missing features
        missing_features = [
            feature for feature in required_features
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Create DataFrame with the same feature structure
        # used during model training
        input_data = pd.DataFrame([{
            "Device Model": data["Device Model"],
            "Operating System": data["Operating System"],
            "App Usage Time (min/day)": float(data["App Usage Time (min/day)"]),
            "Screen On Time (hours/day)": float(data["Screen On Time (hours/day)"]),
            "Battery Drain (mAh/day)": float(data["Battery Drain (mAh/day)"]),
            "Number of Apps Installed": int(data["Number of Apps Installed"]),
            "Data Usage (MB/day)": float(data["Data Usage (MB/day)"]),
            "Age": int(data["Age"]),
            "Gender": data["Gender"]
        }])

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Return prediction
        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Start Flask application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
