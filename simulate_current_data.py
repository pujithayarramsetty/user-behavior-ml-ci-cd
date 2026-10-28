
import argparse
from pathlib import Path

import pandas as pd

DRIFT_DIR = Path("drift")
REFERENCE_FILE = DRIFT_DIR / "reference_data.csv"
CURRENT_FILE = DRIFT_DIR / "current_data.csv"


def simulate_data(mode):
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference data not found. Run prepare_reference.py first."
        )

    reference = pd.read_csv(REFERENCE_FILE)
    current = reference.sample(
        n=len(reference),
        replace=True,
        random_state=42
    ).reset_index(drop=True)

    if mode == "drift":
        current["App Usage Time (min/day)"] = (
            current["App Usage Time (min/day)"] + 300
        )
        current["Screen On Time (hours/day)"] = (
            current["Screen On Time (hours/day)"] + 3
        )

    current.to_csv(CURRENT_FILE, index=False)

    print(f"Simulation mode: {mode}")
    print(f"Rows generated: {len(current)}")
    print(f"Saved to: {CURRENT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stable", "drift"],
        default="stable"
    )
    args = parser.parse_args()

    simulate_data(args.mode)
