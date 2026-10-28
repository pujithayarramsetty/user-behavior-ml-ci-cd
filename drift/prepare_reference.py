
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "user_behavior_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "reference_data.csv"

FEATURES = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
]


def prepare_reference():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    reference = df[FEATURES].dropna()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(OUTPUT_PATH, index=False)

    print("Reference data created successfully!")
    print("Rows:", len(reference))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    prepare_reference()
