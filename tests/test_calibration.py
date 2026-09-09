import unittest

import numpy as np

from emgimu.calibration import find_circular_channel_shift, fit_body_rotation, fit_session_calibration
from emgimu.state import Direction


class CalibrationTests(unittest.TestCase):
    def test_channel_shift_recovers_rotated_profile(self):
        reference = np.arange(1.0, 9.0)
        current = np.roll(reference, 3)
        shift = find_circular_channel_shift(current, reference)
        np.testing.assert_allclose(np.roll(current, shift), reference)

    def test_body_rotation_maps_guided_vectors(self):
        # Device axes are body y, -body x, body z.
        observed = {
            Direction.RIGHT: np.array([0.0, -1.0, 0.0]),
            Direction.FORWARD: np.array([1.0, 0.0, 0.0]),
            Direction.UP: np.array([0.0, 0.0, 1.0]),
        }
        rotation = fit_body_rotation(observed)
        for direction, vector in observed.items():
            mapped = vector @ rotation.T
            expected = {
                Direction.RIGHT: [1, 0, 0], Direction.FORWARD: [0, 1, 0], Direction.UP: [0, 0, 1],
            }[direction]
            np.testing.assert_allclose(mapped, expected, atol=1e-6)

    def test_fit_calibration_is_finite(self):
        rng = np.random.default_rng(1)
        calibration = fit_session_calibration(
            rng.normal(0, 0.01, (200, 8)), rng.normal(0, 1, (600, 8)),
            np.tile([0, 0, 9.81], (200, 1)) + rng.normal(0, 0.01, (200, 3)),
            rng.normal(0, 0.01, (200, 3)),
            rng.normal(0, 1, (600, 3)) + [0, 0, 9.81], rng.normal(0, 1, (600, 3)),
        )
        self.assertTrue(np.isfinite(calibration.transform_emg(np.zeros((2, 8)))).all())


if __name__ == "__main__":
    unittest.main()

