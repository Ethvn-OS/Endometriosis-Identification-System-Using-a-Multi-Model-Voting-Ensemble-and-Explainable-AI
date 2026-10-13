import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_models import (
    ENSEMBLE_NAME,
    MODEL_NAMES,
    align_patient_predictions,
    compute_binary_metrics,
    ensure_output_available,
    stratified_bootstrap_intervals,
    validate_split_request,
    validate_validation_lock,
)


def _patient_frame(probabilities):
    return pd.DataFrame(
        {
            "patient_id": ["n1", "n2", "p1", "p2"],
            "label": [0, 0, 1, 1],
            "source_site": ["MOGaMBO", "MOGaMBO", "UT-D1", "UT-D2"],
            "probability": probabilities,
            "slice_count": [2, 3, 4, 5],
        }
    )


class EvaluateModelsTest(unittest.TestCase):
    def test_test_split_requires_explicit_confirmation(self):
        validate_split_request("validation", confirm_final_test=False)
        validate_split_request("test", confirm_final_test=True)
        with self.assertRaisesRegex(ValueError, "Test evaluation is locked"):
            validate_split_request("test", confirm_final_test=False)

    def test_patient_predictions_are_aligned_and_soft_voted(self):
        predictions = {
            "ResNet50": _patient_frame([0.1, 0.2, 0.7, 0.8]),
            "EfficientNetV2S": _patient_frame([0.2, 0.3, 0.8, 0.9]),
            "DenseNet121": _patient_frame([0.3, 0.4, 0.9, 1.0]),
        }
        patients = align_patient_predictions(predictions)
        expected = np.array([0.2, 0.3, 0.8, 0.9])
        self.assertTrue(
            np.allclose(patients[f"{ENSEMBLE_NAME}_probability"], expected)
        )
        self.assertEqual(
            patients[f"{ENSEMBLE_NAME}_prediction"].tolist(), [0, 0, 1, 1]
        )

    def test_patient_alignment_rejects_label_disagreement(self):
        predictions = {
            name: _patient_frame([0.1, 0.2, 0.7, 0.8]) for name in MODEL_NAMES
        }
        predictions["DenseNet121"].loc[0, "label"] = 1
        with self.assertRaisesRegex(ValueError, "labels do not align"):
            align_patient_predictions(predictions)

    def test_binary_metrics_use_patient_threshold(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.6, 0.4, 0.9])
        metrics = compute_binary_metrics(labels, scores)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)

    def test_bootstrap_intervals_are_deterministic_and_bounded(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
        first = stratified_bootstrap_intervals(labels, scores, iterations=25, seed=7)
        second = stratified_bootstrap_intervals(labels, scores, iterations=25, seed=7)
        self.assertEqual(first, second)
        for interval in first.values():
            self.assertLessEqual(interval["lower"], interval["upper"])
            self.assertGreaterEqual(interval["lower"], 0.0)
            self.assertLessEqual(interval["upper"], 1.0)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ensure_output_available(output)
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                ensure_output_available(output)

    def test_test_lock_rejects_changed_checkpoint(self):
        prior = {
            "split": "validation",
            "threshold": 0.5,
            "checkpoint_sha256": {"ResNet50": "original"},
        }
        current = dict(prior, split="test")
        for key in (
            "bootstrap_iterations",
            "bootstrap_seed",
            "model_names",
            "voting",
            "voting_weights",
            "git_commit",
            "package_versions",
            "evaluation_script_sha256",
            "training_config_sha256",
            "manifest_sha256",
            "release_manifest_sha256",
        ):
            prior[key] = current[key] = "same"
        current["checkpoint_sha256"] = {"ResNet50": "changed"}
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            validate_validation_lock(prior, current)


if __name__ == "__main__":
    unittest.main()
