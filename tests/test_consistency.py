import tempfile
import unittest
from pathlib import Path

from emgimu.consistency import ConditionalConsistencyModel
from emgimu.state import Consistency


class ConsistencyTests(unittest.TestCase):
    def test_fit_predict_and_round_trip(self):
        key = (1, 2, 2, 3)
        records = [(key, 0.6 + index * 0.001, 0.4, 50) for index in range(20)]
        model = ConditionalConsistencyModel(min_samples=12).fit(records)
        status, score = model.predict(key, 0.6, 0.4, 50)
        self.assertEqual(status, Consistency.NORMAL)
        self.assertIsNotNone(score)
        unknown, _ = model.predict((9, 9, 9, 9), 0, 0, 0)
        self.assertEqual(unknown, Consistency.UNKNOWN)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "consistency.json"
            model.save(path)
            loaded = ConditionalConsistencyModel.load(path)
            self.assertIn(key, loaded.stats)


if __name__ == "__main__":
    unittest.main()
