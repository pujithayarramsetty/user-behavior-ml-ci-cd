import json
import sys

# Load model metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

# Minimum acceptable accuracy
MIN_ACCURACY = 0.80

print("Model Accuracy:", accuracy)
print("Required Accuracy:", MIN_ACCURACY)

if accuracy >= MIN_ACCURACY:
    print("QUALITY GATE PASSED")
    sys.exit(0)
else:
    print("QUALITY GATE FAILED")
    sys.exit(1)
