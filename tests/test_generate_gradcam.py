import unittest

import keras
import numpy as np
import pandas as pd

from scripts.evaluate_models import ENSEMBLE_NAME
from scripts.generate_gradcam import (
    EXPECTED_TARGET_LAYERS,
    build_gradcam_models,
    compute_gradcam,
    confusion_category,
    resize_heatmap,
    select_representative_slices,
    validate_recomputed_probability,
    validate_request,
)
from src.modeling import MODEL_NAMES, build_classifier


def _prediction_frames():
    patients = pd.DataFrame(
        {
            "patient_id": ["fn", "fp", "tn1", "tn2", "tp1", "tp2", "tp3"],
            "label": [1, 0, 0, 0, 1, 1, 1],
            "source_site": ["UT-D1", "MOGaMBO", "MOGaMBO", "MOGaMBO", "UT-D1", "UT-D2", "UT-D1"],
            "slice_count": [2] * 7,
            f"{ENSEMBLE_NAME}_probability": [0.4, 0.6, 0.1, 0.3, 0.7, 0.8, 0.9],
            f"{ENSEMBLE_NAME}_prediction": [0, 1, 0, 0, 1, 1, 1],
        }
    )
    for offset, name in enumerate(MODEL_NAMES):
        patients[f"{name}_probability"] = (
            patients[f"{ENSEMBLE_NAME}_probability"] + (offset - 1) * 0.01
        )
        patients[f"{name}_prediction"] = (
            patients[f"{name}_probability"] >= 0.5
        ).astype(int)

    rows = []
    for patient in patients.itertuples(index=False):
        patient_probability = float(
            getattr(patient, f"{ENSEMBLE_NAME}_probability")
        )
        for index, difference in enumerate((-0.02, 0.07)):
            row = {
                "filename": f"test/{patient.patient_id}/slice_{index}.npy",
                "patient_id": patient.patient_id,
                "label": patient.label,
                "source_site": patient.source_site,
                f"{ENSEMBLE_NAME}_probability": patient_probability + difference,
                f"{ENSEMBLE_NAME}_prediction": int(
                    patient_probability + difference >= 0.5
                ),
            }
            for offset, name in enumerate(MODEL_NAMES):
                probability = patient_probability + difference + (offset - 1) * 0.01
                row[f"{name}_probability"] = probability
                row[f"{name}_prediction"] = int(probability >= 0.5)
            rows.append(row)
    return patients, pd.DataFrame(rows)


class GenerateGradcamTest(unittest.TestCase):
    def test_final_test_confirmation_is_required(self):
        validate_request("test", confirm_final_test_explanations=True)
        with self.assertRaisesRegex(ValueError, "locked"):
            validate_request("test", confirm_final_test_explanations=False)

    def test_confusion_categories_are_binary_and_correct(self):
        self.assertEqual(confusion_category(1, 1), "TP")
        self.assertEqual(confusion_category(0, 0), "TN")
        self.assertEqual(confusion_category(0, 1), "FP")
        self.assertEqual(confusion_category(1, 0), "FN")
        with self.assertRaisesRegex(ValueError, "binary"):
            confusion_category(2, 1)

    def test_case_selection_is_deterministic_and_includes_all_errors(self):
        patients, slices = _prediction_frames()
        first = select_representative_slices(patients, slices)
        second = select_representative_slices(patients, slices.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(
            first.set_index("patient_id").loc[["fp", "fn"], "featured"].all()
        )
        featured_correct = first[
            first["featured"] & first["category"].isin(["TP", "TN"])
        ]
        self.assertEqual(featured_correct["category"].value_counts().to_dict(), {"TP": 1, "TN": 1})
        self.assertTrue(first["selected_filename"].str.endswith("slice_0.npy").all())

    def test_native_gradcam_is_finite_and_normalized(self):
        keras.utils.set_random_seed(7)
        inputs = keras.Input((8, 8, 1))
        features = keras.layers.Conv2D(3, 3, activation="relu")(inputs)
        pooled = keras.layers.GlobalAveragePooling2D()(features)
        logits = keras.layers.Dense(1)(pooled)
        model = keras.Model(inputs, [logits, features])
        image = np.random.default_rng(7).random((8, 8, 1), dtype=np.float32)
        positive = compute_gradcam(model, image, target_label=1)
        negative = compute_gradcam(model, image, target_label=0)
        for heatmap in (positive, negative):
            self.assertEqual(heatmap.shape, (6, 6))
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertGreaterEqual(float(heatmap.min()), 0.0)
            self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_heatmap_resize_matches_image_and_remains_bounded(self):
        heatmap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        resized = resize_heatmap(heatmap, (12, 10))
        self.assertEqual(resized.shape, (12, 10))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_small_batch_dependent_probability_drift_is_accepted(self):
        drift = validate_recomputed_probability(
            "ResNet50",
            "UT-D1-027",
            expected_probability=0.72911805,
            actual_probability=0.72991693,
            expected_prediction=1,
        )
        self.assertAlmostEqual(drift, 0.00079888)

    def test_probability_check_rejects_changed_classification(self):
        with self.assertRaisesRegex(ValueError, "classification"):
            validate_recomputed_probability(
                "ResNet50",
                "patient",
                expected_probability=0.501,
                actual_probability=0.499,
                expected_prediction=1,
            )

    def test_probability_check_rejects_large_drift(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            validate_recomputed_probability(
                "ResNet50",
                "patient",
                expected_probability=0.7,
                actual_probability=0.71,
                expected_prediction=1,
            )

    def test_all_classifier_topologies_support_native_gradcam(self):
        sample = np.random.default_rng(8).random(
            (1, 224, 224, 3), dtype=np.float32
        )
        for model_name in MODEL_NAMES:
            with self.subTest(model_name=model_name):
                keras.backend.clear_session()
                model = build_classifier(model_name, weights=None)
                preprocessing, gradcam, target_layer = build_gradcam_models(
                    model, model_name
                )
                prepared = np.asarray(preprocessing(sample, training=False))[0]
                heatmap = compute_gradcam(gradcam, prepared, target_label=1)
                self.assertEqual(target_layer, EXPECTED_TARGET_LAYERS[model_name])
                self.assertEqual(heatmap.shape, (7, 7))
                self.assertTrue(np.isfinite(heatmap).all())


if __name__ == "__main__":
    unittest.main()
