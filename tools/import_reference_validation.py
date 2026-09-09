"""Convert selected legacy HDF5 sessions into isolated 200 Hz validation trials.

This adapter intentionally maps `open_hand` to NEUTRAL because the source
instruction describes a relaxed, loose hand rather than active extension.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from emgimu.calibration import fit_session_calibration
from emgimu.signal import CausalEMGFilter, polyphase_resample
from emgimu.state import Direction, Gesture


SOURCE_TO_TARGET = {
    "2026-09-02_S04": "1",
    "2026-09-02_S05": "3",
    "2026-09-03_S06": "4",
}
DIRECTION_LABELS = {
    "forward": Direction.FORWARD,
    "backward": Direction.BACKWARD,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
    "up": Direction.UP,
    "down": Direction.DOWN,
}
GESTURE_LABELS = {
    "open_hand": Gesture.NEUTRAL,
    "fist": Gesture.FIST,
    "index_pinch": Gesture.PINCH,
}


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def estimate_emg_rate(events: np.ndarray) -> float:
    indices = np.asarray([int(row["sample_index"]) for row in events])
    clocks = np.asarray([int(row["pc_monotonic_ns"]) for row in events])
    valid = (indices >= 0) & (clocks > 0)
    elapsed = (clocks[valid] - clocks[valid][0]) / 1e9
    if np.sum(valid) < 3 or np.ptp(elapsed) <= 0:
        raise ValueError("cannot estimate EMG rate from PC clock")
    rate = float(np.polyfit(elapsed, indices[valid], 1)[0])
    if not 230.0 <= rate <= 270.0:
        raise ValueError(f"unexpected source EMG rate: {rate:.3f} Hz")
    return rate


def align_imu(
    source_indices: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(source_indices, kind="stable")
    sorted_indices = source_indices[order]
    unique_indices, unique_at = np.unique(sorted_indices, return_index=True)
    if len(unique_indices) < 2:
        raise ValueError("not enough IMU samples in trial")
    sorted_accel = accel[order][unique_at]
    sorted_gyro = gyro[order][unique_at]
    aligned_accel = np.column_stack([
        np.interp(target_indices, unique_indices, sorted_accel[:, axis]) for axis in range(3)
    ])
    aligned_gyro = np.column_stack([
        np.interp(target_indices, unique_indices, sorted_gyro[:, axis]) for axis in range(3)
    ])
    return aligned_accel, aligned_gyro


def convert_segment(
    emg: np.ndarray,
    imu_indices: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    start: int,
    end: int,
    source_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if end - start < 40:
        raise ValueError("segment is too short")
    emg_segment = polyphase_resample(emg[start:end], source_rate, 200.0)
    target_indices = np.linspace(start, end - 1, len(emg_segment))
    margin = 4
    mask = (imu_indices >= start - margin) & (imu_indices <= end - 1 + margin)
    aligned_accel, aligned_gyro = align_imu(
        imu_indices[mask], accel[mask], gyro[mask], target_indices,
    )
    return emg_segment, aligned_accel, aligned_gyro


def convert_session(source: Path, output: Path, target_session: str) -> dict[str, object]:
    with h5py.File(source / "session.h5", "r") as handle:
        emg = np.asarray(handle["streams/emg/raw"], dtype=np.float64)
        imu_indices = np.asarray(handle["streams/imu/emg_sample_index"], dtype=np.int64)
        accel = np.asarray(handle["streams/imu/accel"], dtype=np.float64)
        gyro = np.asarray(handle["streams/imu/gyro"], dtype=np.float64)
        trials = np.asarray(handle["trials"])
        source_rate = estimate_emg_rate(np.asarray(handle["events"]))

    output.mkdir(parents=True, exist_ok=True)
    rest_emg: list[np.ndarray] = []
    rest_accel: list[np.ndarray] = []
    rest_gyro: list[np.ndarray] = []
    gesture_emg: list[np.ndarray] = []
    motion_accel: list[np.ndarray] = []
    motion_gyro: list[np.ndarray] = []
    direction_deltas: dict[Direction, list[np.ndarray]] = defaultdict(list)
    written = 0
    for row in trials:
        if not bool(row["valid"]):
            continue
        label = decode(row["label"])
        if label not in DIRECTION_LABELS and label not in GESTURE_LABELS:
            continue
        rest_start = int(row["rest_start_sample"])
        prompt_start = int(row["prompt_start_sample"])
        prompt_end = int(row["prompt_end_sample"])
        rest_values = convert_segment(
            emg, imu_indices, accel, gyro, rest_start, prompt_start, source_rate,
        )
        prompt_values = convert_segment(
            emg, imu_indices, accel, gyro, prompt_start, prompt_end, source_rate,
        )
        rest_emg.append(rest_values[0]); rest_accel.append(rest_values[1]); rest_gyro.append(rest_values[2])
        if label in DIRECTION_LABELS:
            direction = DIRECTION_LABELS[label]
            gesture = Gesture.NEUTRAL
            motion_accel.append(prompt_values[1]); motion_gyro.append(prompt_values[2])
            direction_deltas[direction].append(
                np.mean(prompt_values[1], axis=0) - np.median(rest_values[1], axis=0)
            )
        else:
            direction = Direction.NONE
            gesture = GESTURE_LABELS[label]
            gesture_emg.append(prompt_values[0])
        count = len(prompt_values[0])
        trial_id = f"reference-dong-{source.name}-{int(row['trial_id']):03d}"
        np.savez_compressed(
            output / f"{trial_id}.npz",
            timestamp_ms=np.arange(count, dtype=np.int64) * 5,
            emg=prompt_values[0], accel=prompt_values[1], gyro=prompt_values[2],
            direction=int(direction), gesture=int(gesture),
            stable_mask=np.ones(count, dtype=bool),
            session_id=target_session, trial_id=trial_id,
            session_date=source.name[:10], source_label=label,
        )
        written += 1

    observed = {
        direction: np.median(np.stack(vectors), axis=0)
        for direction, vectors in direction_deltas.items()
    }
    calibration = fit_session_calibration(
        np.concatenate(rest_emg), np.concatenate(gesture_emg),
        np.concatenate(rest_accel), np.concatenate(rest_gyro),
        np.concatenate(motion_accel), np.concatenate(motion_gyro),
        observed_direction_vectors=observed, sample_rate_hz=200,
    )
    calibration_dir = output.parent / "calibration"
    calibration_dir.mkdir(exist_ok=True)
    (calibration_dir / f"session_{target_session}.json").write_text(
        json.dumps(calibration.to_dict(), indent=2), encoding="utf-8",
    )
    return {
        "source": source.name,
        "target_session": target_session,
        "source_emg_rate_hz": source_rate,
        "trials_written": written,
        "notch_50hz": calibration.notch_50hz,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for source_name, target_session in SOURCE_TO_TARGET.items():
        reports.append(convert_session(
            args.source_root / "Dong" / source_name,
            args.output / f"session_{target_session}", target_session,
        ))
    manifest = {
        "purpose": "legacy isolated-action reference validation only",
        "not_official_session4": True,
        "open_hand_mapping": "NEUTRAL because source instruction says relaxed hand",
        "missing_target_gesture": "OPEN",
        "combination_actions_available": False,
        "splits": {"1": "train", "3": "validation", "4": "reference holdout"},
        "sessions": reports,
    }
    (args.output / "REFERENCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
