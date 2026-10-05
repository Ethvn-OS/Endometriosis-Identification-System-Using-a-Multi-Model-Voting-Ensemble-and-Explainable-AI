import unittest

import keras
import numpy as np

from src.modeling import MODEL_NAMES, build_classifier


class ModelingTest(unittest.TestCase):
    def test_all_backbones_accept_unit_interval_inputs(self):
        sample = np.zeros((1, 224, 224, 3), dtype=np.float32)
        for model_name in MODEL_NAMES:
            with self.subTest(model_name=model_name):
                keras.backend.clear_session()
                model = build_classifier(model_name, weights=None)
                prediction = model(sample, training=False).numpy()
                self.assertEqual(prediction.shape, (1, 1))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertGreaterEqual(float(prediction[0, 0]), 0.0)
                self.assertLessEqual(float(prediction[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
