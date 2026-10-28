```python
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


DRIFT_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRIFT_DIR.parent

REPORT_FILE = DRIFT_DIR / "drift_report.json"
LABELED_FILE = DRIFT_DIR / "current_labeled_data.csv"
DATASET_FILE = ROOT_DIR / "user_behavior_dataset.csv"

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
CANDIDATE_ENCODER_FILE = DRIFT_DIR / "candidate_label_encoders.pkl"
METRICS_FILE = DRIFT_DIR / "candidate_metrics.json"

TARGET = "User Behavior Class"
ID_COLUMN = "User ID"

MIN_ACCURACY = 0.85

CATEGORICAL_COLUMNS = [
    "Device Model",
    "Operating System",
    "Gender",
]


def retrain_candidate():
    if not REPORT_FILE.exists():
        raise FileNotFoundError("Drift report is missing.")

    with open(REPORT_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    if not report.get("overall_drift", False):
        print("No drift detected. Retraining skipped.")
        return

    if not DATASET_FILE.exists() or not LABELED_FILE.exists():
        raise FileNotFoundError(
            "Historical dataset or labeled recent data is missing."
        )

    historical = pd.read_csv(DATASET_FILE)
    recent = pd.read_csv(LABELED_FILE)

    required_columns = [
        column for column in historical.columns
        if column not in [ID_COLUMN, TARGET]
    ] + [TARGET]

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
                f"{name} data is missing required columns: {missing}"
            )

    historical = historical[required_columns].copy()
    recent = recent[required_columns].copy()

    combined = pd.concat(
        [historical, recent],
        ignore_index=True
    )

    for column in CATEGORICAL_COLUMNS:
        combined[column] = combined[column].astype(str)

    for column in combined.columns:
        if column not in CATEGORICAL_COLUMNS:
            combined[column] = pd.to_numeric(
                combined[column],
                errors="coerce"
            )

    combined = combined.dropna()

    if combined.empty:
        raise ValueError("No valid labeled training data remains.")

    if combined[TARGET].nunique() < 2:
        raise ValueError("At least two target classes are required.")

    label_encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        combined[column] = encoder.fit_transform(
            combined[column]
        )
        label_encoders[column] = encoder

    X = combined.drop(columns=[TARGET])
    y = combined[TARGET]

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    passes_gate = accuracy >= MIN_ACCURACY

    metrics = {
        "model": "Random Forest",
        "accuracy": float(accuracy),
        "minimum_required_accuracy": MIN_ACCURACY,
        "passes_quality_gate": bool(passes_gate),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": list(X.columns),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    if not passes_gate:
        print("Candidate failed the quality gate.")
        print("Production model remains unchanged.")
        return

    joblib.dump(model, CANDIDATE_FILE)
    joblib.dump(label_encoders, CANDIDATE_ENCODER_FILE)

    print(f"Candidate accuracy: {accuracy:.4f}")
    print(f"Quality gate passed: {passes_gate}")
    print("Candidate model and encoders saved.")
    print("Production model remains unchanged.")


if __name__ == "__main__":
    retrain_candidate()
```
