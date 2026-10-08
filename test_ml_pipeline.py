import unittest
import os
import json


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists("user_behavior_dataset.csv"),
            "Dataset file is missing"
        )

    def test_model_exists(self):
        self.assertTrue(
            os.path.exists("user_behavior_model.pkl"),
            "Trained model is missing"
        )

    def test_metrics_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json"),
            "Metrics file is missing"
        )

    def test_accuracy(self):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.80,
            "Model accuracy is below 80%"
        )


if __name__ == "__main__":
    unittest.main()
