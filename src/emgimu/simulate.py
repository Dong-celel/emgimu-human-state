from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import SessionCalibration
from .state import Direction, Gesture


def create_smoke_dataset(
    output: str | Path,
    *,
    repetitions: int = 1,
    samples_per_trial: int = 120,
    seed: int = 17,
) -> Path:
    """Create non-physiological data for pipeline tests, never for accuracy claims."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    calibration_root = root / "calibration"
    calibration_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    directions = [item for item in Direction if item != Direction.UNKNOWN]
    gestures = [item for item in Gesture if item != Gesture.UNKNOWN]
    t = np.arange(samples_per_trial) / 200.0
    for session in range(1, 5):
        calibration = SessionCalibration.identity()
        (calibration_root / f"session_{session}.json").write_text(
            json.dumps(calibration.to_dict(), indent=2), encoding="utf-8",
        )
        session_root = root / f"session_{session}"
        session_root.mkdir(exist_ok=True)
        for direction in directions:
            for gesture in gestures:
                for repetition in range(repetitions):
                    emg = rng.normal(0, 0.03, (samples_per_trial, 8))
                    # Gesture-specific 30/40 Hz spatial patterns survive the EMG band-pass.
                    primary = int(gesture) * 2
                    emg[:, primary % 8] += (0.6 + 0.08 * session) * np.sin(2 * np.pi * 30 * t)
                    emg[:, (primary + 1) % 8] += 0.35 * np.sin(2 * np.pi * 40 * t)
                    accel = rng.normal(0, 0.015, (samples_per_trial, 3))
                    gyro = rng.normal(0, 0.015, (samples_per_trial, 3))
                    if direction != Direction.NONE:
                        axis = (int(direction) - 1) // 2
                        sign = 1.0 if int(direction) % 2 == 1 else -1.0
                        # Enum pairs are forward/back, left/right, up/down. The exact
                        # synthetic axis is irrelevant; labels remain separable.
                        accel[:, axis] += sign * (0.8 + 0.05 * session)
                        gyro[:, axis] += sign * 0.4 * np.sin(2 * np.pi * 3 * t)
                    trial_id = f"s{session}-d{int(direction)}-h{int(gesture)}-r{repetition}"
                    np.savez(
                        session_root / f"{trial_id}.npz",
                        timestamp_ms=np.arange(samples_per_trial) * 5,
                        emg=emg, accel=accel, gyro=gyro,
                        direction=int(direction), gesture=int(gesture),
                        stable_mask=np.ones(samples_per_trial, dtype=bool),
                        session_id=str(session), trial_id=trial_id,
                        session_date="2099-01-01" if session <= 2 else "2099-01-02",
                    )
    (root / "SMOKE_DATA_ONLY.txt").write_text(
        "Synthetic pipeline data. Do not report its metrics as physiological performance.\n",
        encoding="utf-8",
    )
    return root
