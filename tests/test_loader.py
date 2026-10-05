import unittest

import numpy as np
import pandas as pd

from src.data.loader import assert_disjoint_patients, patient_normalized_weights
from src.evaluation import aggregate_patient_predictions


class LoaderTest(unittest.TestCase):
    def test_patient_weights_equalize_patients_and_classes(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["p1"] * 2 + ["p2"] * 4 + ["n1"] * 3 + ["n2"],
                "label": [1] * 6 + [0] * 4,
            }
        )
        frame["weight"] = patient_normalized_weights(frame)
        patient_totals = frame.groupby(["label", "patient_id"])["weight"].sum()
        class_totals = frame.groupby("label")["weight"].sum()

        for _, totals in patient_totals.groupby(level=0):
            self.assertTrue(np.allclose(totals, totals.iloc[0]))
        self.assertTrue(np.allclose(class_totals, class_totals.iloc[0]))

    def test_leakage_is_rejected_across_any_pair(self):
        frames = {
            "train": pd.DataFrame({"patient_id": ["p1"]}),
            "validation": pd.DataFrame({"patient_id": ["p2"]}),
            "test": pd.DataFrame({"patient_id": ["p1"]}),
        }
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            assert_disjoint_patients(frames)

    def test_predictions_are_mean_pooled_per_patient(self):
        frame = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2"],
                "label": [1, 1, 0],
                "source_site": ["UT-D1", "UT-D1", "MOGaMBO"],
            }
        )
        patients = aggregate_patient_predictions(frame, np.array([0.2, 0.8, 0.1]))
        self.assertAlmostEqual(
            float(patients.loc[patients.patient_id == "p1", "probability"].iloc[0]),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
