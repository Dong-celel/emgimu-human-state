import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

from emgimu.cli import cmd_evaluate, cmd_train_baseline, cmd_validate
from emgimu.simulate import create_smoke_dataset


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class CliSmokeTests(unittest.TestCase):
    def test_dataset_to_model_to_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_smoke_dataset(Path(directory) / "data")
            self.assertEqual(cmd_validate(argparse.Namespace(dataset=str(root), formal=False)), 0)
            model = Path(directory) / "baseline.pkl"
            self.assertEqual(cmd_train_baseline(argparse.Namespace(dataset=str(root), output=str(model))), 0)
            self.assertTrue(model.exists())
            self.assertEqual(cmd_evaluate(argparse.Namespace(
                dataset=str(root), model=str(model), split="validation", unlock_test=False,
            )), 0)

    def test_test_session_requires_explicit_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_smoke_dataset(Path(directory) / "data")
            model = Path(directory) / "baseline.pkl"
            cmd_train_baseline(argparse.Namespace(dataset=str(root), output=str(model)))
            with self.assertRaises(SystemExit):
                cmd_evaluate(argparse.Namespace(
                    dataset=str(root), model=str(model), split="test", unlock_test=False,
                ))


if __name__ == "__main__":
    unittest.main()
