
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# File paths
DRIFT_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRIFT_DIR.parent

REPORT_FILE = DRIFT_DIR / "drift_report.json"
LABELED_FILE = DRIFT_DIR / "current_labeled_data.csv"
DATASET_FILE = ROOT_DIR / "user_behavior_dataset.csv"

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
METRICS_FILE = DRIFT_DIR / "candidate_metrics.json"

# Target column
TARGET = "User Behavior Class"

# Numerical features from the existing dataset
FEATURES = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
]

MIN_ACCURACY = 0.85


def retrain_candidate():

    # 1. Check that a drift report exists
    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            "Drift report missing. Run detect_drift.py first."
        )

    with open(REPORT_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    # 2. Retrain only when drift is detected
    if not report.get("overall_drift", False):
        print("No drift detected. Retraining skipped.")
        return

    # 3. Check for labeled recent data
    if not LABELED_FILE.exists():
        print("Retraining skipped: labeled data file is missing.")
        return

    historical = pd.read_csv(DATASET_FILE)
    recent = pd.read_csv(LABELED_FILE)

    required_columns = FEATURES + [TARGET]

    # 4. Validate dataset columns
    for name, data in [
        ("historical", historical),
        ("recent", recent),
    ]:
        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"{name} data is missing columns: {missing}"
            )

    # 5. Keep required columns and combine labeled data
    historical = historical[required_columns].copy()
    recent = recent[required_columns].copy()

    combined = pd.concat(
        [historical, recent],
        ignore_index=True
    )

    # Convert features to numeric values
    for column in FEATURES:
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
        raise ValueError("No valid labeled training rows found.")

    # 6. Validate target labels
    if combined[TARGET].nunique() < 2:
        raise ValueError(
            "At least two target classes are required for training."
        )

    class_counts = combined[TARGET].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows "
            "for stratified train/test splitting."
        )

    X = combined[FEATURES]
    y = combined[TARGET]

    # 7. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 8. Train candidate model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 9. Evaluate candidate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Candidate accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {MIN_ACCURACY:.2f}")

    # 10. Save candidate metrics
    metrics = {
        "accuracy": float(accuracy),
        "minimum_required_accuracy": MIN_ACCURACY,
        "passes_quality_gate": bool(accuracy >= MIN_ACCURACY),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    # 11. Save candidate only if it passes the quality gate
    if accuracy < MIN_ACCURACY:
        print("Candidate failed the quality gate.")
        print("The existing production model remains unchanged.")
        return

    joblib.dump(model, CANDIDATE_FILE)

    print(f"Candidate model saved: {CANDIDATE_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")
    print("Candidate passed the accuracy quality gate.")
    print("The production model has not been replaced.")


if __name__ == "__main__":
    retrain_candidate()
