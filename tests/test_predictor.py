"""The pure-Python predictor must reproduce the saved baseline probabilities exactly."""

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import predictor  # noqa: E402

APP_DATA = ROOT / "app" / "data"
TOLERANCE = 1e-9


def _load_rows() -> list[dict]:
    with open(APP_DATA / "test_customers.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class PredictorReproducesBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(APP_DATA / "baseline_model.json", encoding="utf-8") as fh:
            cls.spec = json.load(fh)
        cls.rows = _load_rows()

    def test_has_full_test_set(self) -> None:
        self.assertEqual(len(self.rows), 1409)

    def test_probabilities_match_within_tolerance(self) -> None:
        max_diff = 0.0
        for row in self.rows:
            prob = predictor.predict_proba(self.spec, row)
            diff = abs(prob - float(row["baseline_probability"]))
            max_diff = max(max_diff, diff)
            self.assertLess(diff, TOLERANCE, f"{row['customerID']}: {prob} vs {row['baseline_probability']}")
        print(f"\nmax |predictor - baseline_probability| over {len(self.rows)} rows: {max_diff:.2e}")

    def test_labels_match_at_threshold(self) -> None:
        threshold = self.spec["threshold"]
        for row in self.rows:
            label = "Yes" if predictor.predict_proba(self.spec, row) >= threshold else "No"
            self.assertEqual(label, row["baseline_predicted"], row["customerID"])

    def test_missing_total_charges_uses_median(self) -> None:
        blanks = [r for r in self.rows if r["TotalCharges"] == ""]
        self.assertGreater(len(blanks), 0, "expected some customers with blank TotalCharges")
        for row in blanks:
            diff = abs(predictor.predict_proba(self.spec, row) - float(row["baseline_probability"]))
            self.assertLess(diff, TOLERANCE, row["customerID"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
