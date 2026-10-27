@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a valid JSON object"
            }), 400

        # Get the exact feature order used during model training.
        feature_names = list(model.feature_names_in_)

        # Check for missing features.
        missing_features = [
            feature for feature in feature_names
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features,
                "expected_features": feature_names
            }), 400

        # Build input in the exact order expected by the model.
        processed_data = {}

        for column in feature_names:
            value = data[column]

            if column in label_encoders:
                value = str(value)
                encoder = label_encoders[column]

                if value not in [str(item) for item in encoder.classes_]:
                    return jsonify({
                        "error": f"Unknown value for {column}",
                        "received_value": value,
                        "allowed_values": [
                            str(item) for item in encoder.classes_
                        ]
                    }), 400

                value = encoder.transform([value])[0]

            processed_data[column] = value

        input_df = pd.DataFrame(
            [processed_data],
            columns=feature_names
        )

        prediction = model.predict(input_df)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        logger.info("Prediction completed successfully")

        return jsonify({
            "prediction": prediction,
            "message": "Prediction successful"
        }), 200

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check application logs."
        }), 500
