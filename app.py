from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("user_behavior_model.pkl")

# Load the encoders created during model training
label_encoders = joblib.load("label_encoders.pkl")


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok"
    })


# Health check endpoint for Docker
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

        # Check empty request
        if not data:
            return jsonify({
                "error": "No JSON data provided"
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

        # Find missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required features",
                "missing_fields": missing_fields
            }), 400

        # Create input DataFrame
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

        # Encode categorical values using the same encoders
        # used during model training
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
                return jsonify({
                    "error": f"Unknown value for {column}"
                }), 400

        # Make prediction
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
