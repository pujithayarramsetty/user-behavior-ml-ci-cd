
from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

DRIFT_DIR = Path(__file__).resolve().parent
REPORT_FILE = DRIFT_DIR / "drift_report.json"
LABELED_FILE = DRIFT_DIR / "current_labeled_data.csv"
DATASET_FILE = Path("user_behavior_dataset.csv")
CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"

TARGET = "User Behavior Class"

FEATURES = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
]


def retrain_candidate():
    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            "Drift report missing. Run detect_drift.py first."
        )

    report = json.loads(REPORT_FILE.read_text())

    if not report.get("overall_drift", False):
        print("No drift detected. Retraining skipped.")
        return

    if not LABELED_FILE.exists():
        print(
            "Retraining skipped: current_labeled_data.csv is missing. "
            "New data must have verified target labels."
        )
        return

    historical = pd.read_csv(DATASET_FILE)
    recent = pd.read_csv(LABELED_FILE)

    required = FEATURES + [TARGET]

    for name, data in [("historical", historical), ("recent", recent)]:
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"{name} data missing columns: {missing}")

    combined = pd.concat(
        [historical[required], recent[required]],
        ignore_index=True
    ).dropna()

    X = combined[FEATURES]
    y = combined[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, CANDIDATE_FILE)

metrics = {
    "accuracy": float(accuracy)
}

(DRIFT_DIR / "candidate_metrics.json").write_text(
    json.dumps(metrics, indent=4)
)

    print(f"Candidate model saved: {CANDIDATE_FILE}")
    print(f"Candidate accuracy: {accuracy:.4f}")
    print("The existing model has not been replaced.")


if __name__ == "__main__":
    retrain_candidate()
