
from pathlib import Path
import json

import pandas as pd
from scipy.stats import ks_2samp

DRIFT_DIR = Path("drift")
REFERENCE_FILE = DRIFT_DIR / "reference_data.csv"
CURRENT_FILE = DRIFT_DIR / "current_data.csv"
REPORT_FILE = DRIFT_DIR / "drift_report.json"

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2


def detect_drift():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference data missing. Prepare reference data first."
        )

    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            "Current data missing. Simulate current data first."
        )

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    if list(reference.columns) != list(current.columns):
        raise ValueError("Reference and current features do not match.")

    feature_results = {}

    for feature in reference.columns:
        result = ks_2samp(
            reference[feature].dropna(),
            current[feature].dropna()
        )

        drift_detected = result.pvalue < P_VALUE_THRESHOLD

        feature_results[feature] = {
            "p_value": float(result.pvalue),
            "drift_detected": bool(drift_detected)
        }

        print(
            f"{feature}: p-value={result.pvalue:.4f}, "
            f"drift={drift_detected}"
        )

    drifted_count = sum(
        result["drift_detected"]
        for result in feature_results.values()
    )

    report = {
        "method": "Kolmogorov-Smirnov test",
        "p_value_threshold": P_VALUE_THRESHOLD,
        "minimum_drifted_features": MIN_DRIFTED_FEATURES,
        "drifted_features_count": drifted_count,
        "overall_drift": drifted_count >= MIN_DRIFTED_FEATURES,
        "features": feature_results
    }

    REPORT_FILE.write_text(json.dumps(report, indent=4))

    print("\nOverall drift:", report["overall_drift"])
    print("Report saved to:", REPORT_FILE)


if __name__ == "__main__":
    detect_drift()
