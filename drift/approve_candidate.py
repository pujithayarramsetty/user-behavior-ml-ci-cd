
from pathlib import Path
import json

import joblib

DRIFT_DIR = Path(__file__).resolve().parent
ROOT_DIR = DRIFT_DIR.parent

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
METRICS_FILE = DRIFT_DIR / "candidate_metrics.json"

PRODUCTION_MODEL = ROOT_DIR / "user_behavior_model.pkl"
MIN_ACCURACY = 0.85


def approve_candidate():
    if not CANDIDATE_FILE.exists():
        print("Candidate model not found. Nothing to approve.")
        return

    if not METRICS_FILE.exists():
        print("Candidate metrics not found. Approval denied.")
        return

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy")

    if accuracy is None or accuracy < MIN_ACCURACY:
        print("Candidate failed the quality gate.")
        print("Existing production model remains unchanged.")
        return

    # Validate the candidate model before promotion
    candidate = joblib.load(CANDIDATE_FILE)

    # Back up the existing production model if present
    if PRODUCTION_MODEL.exists():
        backup_file = ROOT_DIR / "user_behavior_model.backup.pkl"
        import shutil
        shutil.copy2(PRODUCTION_MODEL, backup_file)
        print(f"Production model backed up to: {backup_file}")

    # Promote the candidate model
    joblib.dump(candidate, PRODUCTION_MODEL)

    print(f"Candidate approved. Accuracy: {accuracy:.4f}")
    print(f"Production model saved to: {PRODUCTION_MODEL}")


if __name__ == "__main__":
    approve_candidate()
