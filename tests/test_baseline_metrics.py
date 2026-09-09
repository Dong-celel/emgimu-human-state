import importlib.util
import unittest

import numpy as np

from emgimu.baseline import BaselinePredictor
from emgimu.metrics import evaluate_predictions, first_stable_latency
from emgimu.state import Direction, Gesture


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class BaselineMetricsTests(unittest.TestCase):
    def test_svm_fit_calibrate_reject_and_predict(self):
        rng = np.random.default_rng(4)
        train_count = 240
        validation_count = 120
        train_d = np.arange(train_count) % 7
        train_h = np.arange(train_count) % 4
        validation_d = np.arange(validation_count) % 7
        validation_h = np.arange(validation_count) % 4
        train_imu = rng.normal(0, 0.05, (train_count, 36))
        train_emg = rng.normal(0, 0.05, (train_count, 48))
        validation_imu = rng.normal(0, 0.05, (validation_count, 36))
        validation_emg = rng.normal(0, 0.05, (validation_count, 48))
        train_imu[np.arange(train_count), train_d] += 3
        train_emg[np.arange(train_count), train_h] += 3
        validation_imu[np.arange(validation_count), validation_d] += 3
        validation_emg[np.arange(validation_count), validation_h] += 3
        predictor = BaselinePredictor().fit(
            train_emg, train_imu, train_d, train_h,
            validation_emg_features=validation_emg,
            validation_imu_features=validation_imu,
            validation_direction=validation_d,
            validation_gesture=validation_h,
        )
        prediction = predictor.predict_features(validation_emg[10], validation_imu[10])
        self.assertEqual(prediction.direction, Direction(validation_d[10]))
        self.assertEqual(prediction.gesture, Gesture(validation_h[10]))

    def test_unknown_counts_as_error(self):
        result = evaluate_predictions(
            np.array([0, 1]), np.array([0, 1]),
            np.array([0, -1]), np.array([0, -1]),
            np.array([0.9, 0.2]), np.array([0.9, 0.2]),
            latency_ms=np.array([200, 280]),
        )
        self.assertLess(result.direction_macro_f1, 1)
        self.assertEqual(result.joint_accuracy, 0.5)
        self.assertEqual(result.p50_latency_ms, 240)
        self.assertTrue(result.p95_latency_ms <= 300)
        self.assertEqual(result.direction_labels, (-1, 0, 1))

    def test_latency_matches_first_correct_stable_state(self):
        latency = first_stable_latency(100, np.array([80, 120, 160]), np.array([0, 0, 2]), 2)
        self.assertEqual(latency, 60)


if __name__ == "__main__":
    unittest.main()
