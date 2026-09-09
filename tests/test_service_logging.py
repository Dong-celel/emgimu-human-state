import csv
import tempfile
import unittest
from pathlib import Path

from emgimu.service import StateCsvLogger
from emgimu.state import (
    Confidence, Consistency, Direction, Gesture, HumanState, Phase, PhasePair,
    SignalQuality,
)


class ServiceLoggingTests(unittest.TestCase):
    def test_csv_is_fit_consistency_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv"
            with StateCsvLogger(path) as logger:
                logger.write(HumanState(
                    120, Direction.UNKNOWN, Gesture.UNKNOWN, 0.2,
                    PhasePair(Phase.UNKNOWN, Phase.UNKNOWN), 0.1,
                    Consistency.UNKNOWN, Confidence(0.1, 0.2, 0.0, 0.0),
                    SignalQuality(0.8, 0.9),
                ), -40.0)
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["timestamp_ms"], "120")
            self.assertEqual(row["onset_lag_ms"], "-40.0")
            self.assertIn("activation", row)


if __name__ == "__main__":
    unittest.main()
