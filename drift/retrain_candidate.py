
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# File paths
# --------------------------------------------------

DRIFT_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRIFT_DIR.parent

REPORT_FILE = DRIFT_DIR / "drift_report.json"
LABELED_FILE = DRIFT_DIR / "current_labeled_data.csv"
DATASET_FILE = ROOT_DIR / "user_behavior_dataset.csv"

PRODUCTION_MODEL = ROOT_DIR / "user_behavior_model.pkl"
PRODUCTION_ENCODERS = ROOT_DIR / "label_encoders.pkl"

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
CANDIDATE_ENCODER_FILE = DRIFT_DIR / "candidate_label_encoders.pkl"
METRICS_FILE = DRIFT_DIR / "candidate_metrics.json"

TARGET = "User Behavior Class"
ID_COLUMN = "User ID"
MIN_ACCURACY = 0.85


# --------------------------------------------------
# Retrain candidate model
# --------------------------------------------------

def retrain_candidate():

    # 1. Check for drift report
    if not REPORT_FILE.exists():
        raise FileNotFoundError("Drift report is missing.")

    with open(REPORT_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    if not report.get("overall_drift", False):
        print("No drift detected. Retraining skipped.")
        return

    # 2. Check required files
    required_files = [
        DATASET_FILE,
        LABELED_FILE,
        PRODUCTION_MODEL,
        PRODUCTION_ENCODERS,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file missing: {file_path.name}"
            )

    # 3. Load production model and encoders
    production_model = joblib.load(PRODUCTION_MODEL)
    production_encoders = joblib.load(PRODUCTION_ENCODERS)

    feature_names = list(production_model.feature_names_in_)

    # 4. Load historical and labeled data
    historical = pd.read_csv(DATASET_FILE)
    recent = pd.read_csv(LABELED_FILE)

    required_columns = feature_names + [TARGET]

    for name, data in [
        ("historical", historical),
        ("recent", recent),
    ]:
        missing = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"{name} data is missing columns: {missing}"
            )

    # 5. Select features in production order
    historical = historical[required_columns].copy()
    recent = recent[required_columns].copy()

    combined = pd.concat(
        [historical, recent],
        ignore_index=True
    )

    # 6. Encode categorical features using production encoders
    for column, encoder in production_encoders.items():

        if column not in feature_names:
            continue

        values = combined[column].astype(str)
        known_values = set(str(value) for value in encoder.classes_)

        unknown_values = sorted(set(values) - known_values)

        if unknown_values:
            raise ValueError(
                f"Unknown values in {column}: {unknown_values}. "
                "Production encoder cannot safely encode them."
            )

        combined[column] = encoder.transform(values)

    # 7. Convert remaining features and target to numeric
    for column in feature_names:
        if column not in production_encoders:
            combined[column] = pd.to_numeric(
                combined[column],
                errors="coerce"
            )

    combined[TARGET] = pd.to_numeric(
        combined[TARGET],
        errors="coerce"
    )

    combined = combined.dropna(subset=required_columns)

    if combined.empty:
        raise ValueError("No valid labeled training rows remain.")

    if combined[TARGET].nunique() < 2:
        raise ValueError("At least two target classes are required.")

    if combined[TARGET].value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two rows."
        )

    # 8. Prepare training data in the exact production feature order
    X = combined[feature_names]
    y = combined[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 9. Train candidate model
    candidate = production_model.__class__(
        n_estimators=100,
        random_state=42,
    )

    candidate.fit(X_train, y_train)

    # 10. Evaluate candidate
    predictions = candidate.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    passes_gate = accuracy >= MIN_ACCURACY

    metrics = {
        "model": candidate.__class__.__name__,
        "accuracy": float(accuracy),
        "minimum_required_accuracy": MIN_ACCURACY,
        "passes_quality_gate": bool(passes_gate),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": feature_names,
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print(f"Candidate accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {MIN_ACCURACY:.2f}")
    print(f"Quality gate passed: {passes_gate}")

    # 11. Save candidate only if it passes
    if not passes_gate:
        print("Candidate failed the quality gate.")
        print("Production model remains unchanged.")
        return

    joblib.dump(candidate, CANDIDATE_FILE)
    joblib.dump(production_encoders, CANDIDATE_ENCODER_FILE)

    print("Candidate model saved successfully.")
    print("Candidate encoders saved successfully.")
    print("Production model remains unchanged.")


if __name__ == "__main__":
    retrain_candidate()
