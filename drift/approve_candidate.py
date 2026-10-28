
from pathlib import Path
import json
import joblib

DRIFT_DIR = Path(__file__).resolve().parent

CANDIDATE_FILE = DRIFT_DIR / "candidate_model.pkl"
REPORT_FILE = DRIFT_DIR / "drift_report.json"
ACCURACY_FILE = DRIFT_DIR / "candidate_metrics.json"

# Existing production model
PRODUCTION_MODEL = Path("user_behavior_model.pkl")

MIN_ACCURACY = 0.85


def approve_candidate():
    if not CANDIDATE_FILE.exists():
        print("Candidate model not found. Nothing to approve.")
        return

    if not ACCURACY_FILE.exists():
        print("Candidate metrics missing. Approval denied.")
        return

    metrics = json.loads(ACCURACY_FILE.read_text())
    accuracy = metrics.get("accuracy")

    if accuracy is None or accuracy < MIN_ACCURACY:
        print("Candidate failed the accuracy requirement.")
        print("Existing production model remains unchanged.")
        return

    # Keep the original model safe before promotion.
    if PRODUCTION_MODEL.exists():
        backup = PRODUCTION_MODEL.with_suffix(".backup.pkl")
        joblib.copy if False else None
        import shutil
        shutil.copy2(PRODUCTION_MODEL, backup)
        print(f"Existing model backed up to: {backup}")

    joblib.dump(joblib.load(CANDIDATE_FILE), PRODUCTION_MODEL)

    print(f"Candidate approved with accuracy: {accuracy:.4f}")
    print("Candidate promoted to production model.")


if __name__ == "__main__":
    approve_candidate()
