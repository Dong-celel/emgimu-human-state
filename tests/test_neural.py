import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch neural extra is not installed")
class NeuralTests(unittest.TestCase):
    def test_shapes_and_causality(self):
        import torch
        from emgimu.neural import DualBranchCausalNet

        torch.manual_seed(0)
        model = DualBranchCausalNet().eval()
        emg = torch.randn(2, 40, 8)
        imu = torch.randn(2, 40, 6)
        output = model(emg, imu)
        self.assertEqual(tuple(output["direction"].shape), (2, 7))
        self.assertEqual(tuple(output["gesture"].shape), (2, 4))

        # Editing the future cannot affect an earlier prefix output.
        prefix_emg = emg[:, :30].clone(); prefix_imu = imu[:, :30].clone()
        first = model(prefix_emg, prefix_imu)["gesture"]
        longer_emg = torch.cat([prefix_emg, torch.randn(2, 10, 8)], dim=1)
        longer_imu = torch.cat([prefix_imu, torch.randn(2, 10, 6)], dim=1)
        second = model(longer_emg[:, :30], longer_imu[:, :30])["gesture"]
        torch.testing.assert_close(first, second)

    def test_short_training_and_artifact_round_trip(self):
        import tempfile
        from pathlib import Path

        from emgimu.data import RawWindows
        from emgimu.neural import DualBranchConfig
        from emgimu.training import load_neural_artifact, save_neural_artifact, train_dual_branch

        rng = np.random.default_rng(9)

        def windows(count):
            direction = np.arange(count) % 7
            gesture = np.arange(count) % 4
            emg = rng.normal(0, 0.1, (count, 40, 8)).astype("float32")
            imu = rng.normal(0, 0.1, (count, 40, 6)).astype("float32")
            for index in range(count):
                emg[index, :, gesture[index]] += 1.0
                imu[index, :, direction[index] % 6] += 1.0
            return RawWindows(
                emg, imu, direction.astype("int64"), gesture.astype("int64"),
                np.asarray([f"t{i}" for i in range(count)]), np.arange(count),
            )

        result = train_dual_branch(
            windows(56), windows(28),
            config=DualBranchConfig(hidden_dim=8, embedding_dim=16),
            max_epochs=2, patience=2, batch_size=28,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_neural_artifact(result, path)
            predictor = load_neural_artifact(path)
            prediction = predictor.predict(np.zeros((40, 8)), np.zeros((40, 6)))
            self.assertGreaterEqual(prediction.q_direction, 0)


if __name__ == "__main__":
    unittest.main()
