import tempfile
import unittest
from pathlib import Path

import numpy as np

from emgimu.data import DatasetError, dataset_report, discover_trials, validate_no_leakage


def write_trial(path: Path, session: str, trial_id: str, direction: int = 0, gesture: int = 0):
    n = 80
    rng = np.random.default_rng(abs(hash((session, trial_id))) % (2**32))
    np.savez(
        path, timestamp_ms=np.arange(n) * 5, emg=rng.normal(size=(n, 8)),
        accel=rng.normal(size=(n, 3)), gyro=rng.normal(size=(n, 3)),
        direction=direction, gesture=gesture, session_id=session, trial_id=trial_id,
    )


class DataTests(unittest.TestCase):
    def test_discovers_and_reports_missing_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trial(root / "a.npz", "1", "trial-a")
            trials = discover_trials(root)
            report = dataset_report(trials)
            self.assertEqual(report["split_counts"]["train"], 1)
            self.assertEqual(len(report["missing_combinations"]), 27)
            self.assertEqual(len(report["missing_combinations_by_split"]["test"]), 28)

    def test_rejects_trial_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trial(root / "a.npz", "1", "same")
            write_trial(root / "b.npz", "3", "same")
            with self.assertRaises(DatasetError):
                discover_trials(root)

    def test_rejects_duplicate_trial_within_same_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trial(root / "a.npz", "1", "same")
            write_trial(root / "b.npz", "2", "same")
            with self.assertRaises(DatasetError):
                discover_trials(root)


if __name__ == "__main__":
    unittest.main()
