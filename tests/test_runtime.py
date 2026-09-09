import importlib.util
import unittest

import numpy as np

from emgimu.baseline import HeadPrediction
from emgimu.calibration import SessionCalibration
from emgimu.runtime import HumanStateEstimator
from emgimu.state import Direction, Gesture, Phase, QualityFlag


class FixedPredictor:
    def predict(self, normalized_emg, normalized_imu):
        return HeadPrediction(Direction.RIGHT, Gesture.FIST, 0.95, 0.90)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is required by runtime filtering")
class RuntimeTests(unittest.TestCase):
    def test_causal_output_rate_and_hysteresis(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        rng = np.random.default_rng(3)
        states = estimator.push_batch(
            np.arange(80) * 5,
            rng.normal(size=(80, 8)),
            rng.normal(size=(80, 3)),
            rng.normal(size=(80, 3)),
        )
        self.assertEqual(len(states), 6)  # at sample 40, then every 8 samples
        self.assertEqual(states[0].direction, Direction.NONE)
        self.assertEqual(states[0].phase.arm, Phase.ONSET)
        self.assertEqual(states[1].direction, Direction.RIGHT)
        self.assertTrue(states[-1].signal_quality.flags & QualityFlag.CALIBRATED)

    def test_out_of_order_sample_is_ignored(self):
        estimator = HumanStateEstimator(FixedPredictor(), SessionCalibration.identity())
        sample = np.ones(8)
        self.assertIsNone(estimator.push_sample(10, sample, np.ones(3), np.ones(3)))
        self.assertIsNone(estimator.push_sample(9, sample, np.ones(3), np.ones(3)))
        self.assertEqual(estimator.last_timestamp_ms, 10)


if __name__ == "__main__":
    unittest.main()
