import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_endpoint(self):
        response = self.client.post(
            "/predict",
            json={
                "Device Model": "Google Pixel 5",
                "Operating System": "Android",
                "App Usage Time (min/day)": 393,
                "Screen On Time (hours/day)": 6.4,
                "Battery Drain (mAh/day)": 1872,
                "Number of Apps Installed": 67,
                "Data Usage (MB/day)": 1122,
                "Age": 40,
                "Gender": "Male"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Device Model": "Google Pixel 5",
                "Operating System": "Android"
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )

    def test_empty_request_validation(self):
        response = self.client.post(
            "/predict",
            json={}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "error",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
