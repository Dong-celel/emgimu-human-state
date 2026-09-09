import importlib.util
import unittest

import numpy as np

from emgimu.external import adapt_external_emg
from emgimu.signal import extract_emg_features, extract_imu_features, polyphase_resample


class SignalExternalTests(unittest.TestCase):
    def test_feature_dimensions(self):
        rng = np.random.default_rng(2)
        self.assertEqual(extract_emg_features(rng.normal(size=(40, 8))).shape, (48,))
        self.assertEqual(extract_imu_features(rng.normal(size=(40, 6))).shape, (36,))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is optional in lightweight runtime")
    def test_resampling_applies_expected_ratio(self):
        values = np.arange(1000, dtype=float)[:, None]
        output = polyphase_resample(values, 1000, 200)
        self.assertEqual(output.shape, (200, 1))

    def test_db5_splits_two_real_armbands(self):
        values = np.zeros((10, 16))
        parts = adapt_external_emg("ninapro_db5", values)
        self.assertEqual([part.shape for part in parts], [(10, 8), (10, 8)])

    def test_noncommercial_license_is_enforced(self):
        with self.assertRaises(PermissionError):
            adapt_external_emg("meta_gni", np.zeros((10, 16)), commercial_use=True)


if __name__ == "__main__":
    unittest.main()
