
from pathlib import Path
import json
import shutil

import joblib

DRIFT_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRIFT_DIR.parent

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
CANDIDATE_ENCODER_FILE = DRIFT_DIR / "candidate_label_encoders.pkl"
METRICS_FILE = DRIFT_DIR / "candidate_metrics.json"

PRODUCTION_MODEL = ROOT_DIR / "user_behavior_model.pkl"
PRODUCTION_ENCODERS = ROOT_DIR / "label_encoders.pkl"

MIN_ACCURACY = 0.85


def approve_candidate():
    required_files = [
        CANDIDATE_FILE,
        CANDIDATE_ENCODER_FILE,
        METRICS_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            print(f"Approval denied: {file_path.name} is missing.")
            return

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy")
    passes_gate = metrics.get("passes_quality_gate", False)

    if (
        accuracy is None
        or accuracy < MIN_ACCURACY
        or not passes_gate
    ):
        print("Candidate failed the quality gate.")
        print("Production files remain unchanged.")
        return

    candidate = joblib.load(CANDIDATE_FILE)
    candidate_encoders = joblib.load(CANDIDATE_ENCODER_FILE)

    # Check that the candidate uses the same features as production.
    production = joblib.load(PRODUCTION_MODEL)
    production_features = list(production.feature_names_in_)
    candidate_features = list(candidate.feature_names_in_)

    if candidate_features != production_features:
        print("Approval denied: candidate features do not match production.")
        print("Production files remain unchanged.")
        return

    expected_encoders = joblib.load(PRODUCTION_ENCODERS)

    if set(candidate_encoders.keys()) != set(expected_encoders.keys()):
        print("Approval denied: encoder columns do not match production.")
        print("Production files remain unchanged.")
        return

    # Back up both production files.
    backup_model = ROOT_DIR / "user_behavior_model.backup.pkl"
    backup_encoders = ROOT_DIR / "label_encoders.backup.pkl"

    shutil.copy2(PRODUCTION_MODEL, backup_model)
    shutil.copy2(PRODUCTION_ENCODERS, backup_encoders)

    # Save approved candidate files.
    joblib.dump(candidate, PRODUCTION_MODEL)
    joblib.dump(candidate_encoders, PRODUCTION_ENCODERS)

    print(f"Candidate approved. Accuracy: {accuracy:.4f}")
    print("Production model and encoders updated successfully.")


if __name__ == "__main__":
    approve_candidate()
