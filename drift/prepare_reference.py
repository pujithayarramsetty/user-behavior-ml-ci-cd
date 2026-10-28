import pandas as pd
from pathlib import Path

# Load the existing User Behavior dataset
DATASET_PATH = Path("user_behavior_dataset.csv")

FEATURES = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
]

# Check whether the dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "user_behavior_dataset.csv was not found."
    )

# Read dataset
df = pd.read_csv(DATASET_PATH)

# Check required columns
missing = [column for column in FEATURES if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep reference features and remove missing values
reference_data = df[FEATURES].dropna()

# Create drift folder
output_path = Path("drift")
output_path.mkdir(exist_ok=True)

# Save reference data
reference_data.to_csv(
    output_path / "reference_data.csv",
    index=False
)

print("Reference data prepared successfully!")
print("Rows:", len(reference_data))
print("Features:", FEATURES)
print("Saved to: drift/reference_data.csv")
