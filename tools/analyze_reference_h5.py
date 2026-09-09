"""Summarize candidate sessions without modifying source recordings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def summarize(path: Path, root: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        emg_index = np.asarray(handle["streams/emg/sample_index"])
        imu_index = np.asarray(handle["streams/imu/emg_sample_index"])
        emg = np.asarray(handle["streams/emg/raw"])
        accel = np.asarray(handle["streams/imu/accel"])
        gyro = np.asarray(handle["streams/imu/gyro"])
        trials = np.asarray(handle["trials"])
        events = np.asarray(handle["events"])
        labels = Counter(text(row["label"]) for row in trials if bool(row["valid"]))
        invalid = Counter(text(row["reject_reason"]) for row in trials if not bool(row["valid"]))
        event_times = np.asarray([float(row["time_sec"]) for row in events])
        event_indices = np.asarray([int(row["sample_index"]) for row in events])
        event_pc_ns = np.asarray([int(row["pc_monotonic_ns"]) for row in events])
        valid_time = np.isfinite(event_times) & (event_indices >= 0)
        slope = None
        if np.sum(valid_time) >= 2 and np.ptp(event_times[valid_time]) > 0:
            slope = float(np.polyfit(event_times[valid_time], event_indices[valid_time], 1)[0])
        pc_valid = (event_indices >= 0) & (event_pc_ns > 0)
        pc_rate = None
        if np.sum(pc_valid) >= 2:
            pc_seconds = (event_pc_ns[pc_valid] - event_pc_ns[pc_valid][0]) / 1e9
            if np.ptp(pc_seconds) > 0:
                pc_rate = float(np.polyfit(pc_seconds, event_indices[pc_valid], 1)[0])
        imu_pc_ns = np.asarray(handle["streams/imu/pc_monotonic_ns"])
        imu_pc_rate = None
        imu_positive = imu_pc_ns > 0
        if np.sum(imu_positive) >= 2:
            duration = (imu_pc_ns[imu_positive][-1] - imu_pc_ns[imu_positive][0]) / 1e9
            if duration > 0:
                imu_pc_rate = float((np.sum(imu_positive) - 1) / duration)
        def diff_stats(values: np.ndarray) -> dict[str, float]:
            differences = np.diff(values.astype(np.int64))
            return {
                "median": float(np.median(differences)),
                "p95": float(np.percentile(differences, 95)),
                "nonpositive": int(np.sum(differences <= 0)),
                "gaps_gt_1": int(np.sum(differences > 1)),
            }
        return {
            "file": str(path.relative_to(root)),
            "valid_trials": int(sum(labels.values())),
            "invalid_trials": int(len(trials) - sum(labels.values())),
            "labels": dict(sorted(labels.items())),
            "invalid_reasons": dict(sorted(invalid.items())),
            "emg_samples": len(emg),
            "imu_samples": len(accel),
            "event_index_rate_hz": slope,
            "emg_rate_from_pc_clock_hz": pc_rate,
            "imu_rate_from_pc_clock_hz": imu_pc_rate,
            "emg_index_range": [int(emg_index[0]), int(emg_index[-1])],
            "imu_emg_index_range": [int(np.min(imu_index)), int(np.max(imu_index))],
            "event_sample_index_range": [int(np.min(event_indices)), int(np.max(event_indices))],
            "event_time_range_sec": [float(np.min(event_times)), float(np.max(event_times))],
            "trial_sample_range": [
                int(min(row["trial_start_sample"] for row in trials)),
                int(max(row["trial_end_sample"] for row in trials)),
            ],
            "emg_index_diff": diff_stats(emg_index),
            "imu_emg_index_diff": diff_stats(imu_index),
            "emg_nan": int(np.sum(~np.isfinite(emg))),
            "imu_nan": int(np.sum(~np.isfinite(accel))) + int(np.sum(~np.isfinite(gyro))),
            "emg_saturated_fraction": float(np.mean(
                (emg == np.iinfo(emg.dtype).min) | (emg == np.iinfo(emg.dtype).max)
            )),
            "accel_abs_p99": float(np.percentile(np.abs(accel), 99)),
            "gyro_abs_p99": float(np.percentile(np.abs(gyro), 99)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps([
        summarize(path, args.root) for path in sorted(args.root.rglob("session.h5"))
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
