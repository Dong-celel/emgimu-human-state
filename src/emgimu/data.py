from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from .calibration import SessionCalibration
from .signal import CausalEMGFilter, extract_emg_features, extract_imu_features
from .state import Direction, Gesture


SESSION_SPLITS = {
    "1": "train",
    "2": "train",
    "3": "validation",
    "4": "test",
}

EVENT_FIELDS = ("cue_ms", "arm_onset_ms", "arm_end_ms", "hand_onset_ms", "hand_end_ms")


@dataclass(frozen=True, slots=True)
class Trial:
    path: Path
    session_id: str
    trial_id: str
    timestamp_ms: np.ndarray
    emg: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    direction: np.ndarray
    gesture: np.ndarray
    stable_mask: np.ndarray
    session_date: str | None
    events: dict[str, float]

    @property
    def split(self) -> str:
        try:
            return SESSION_SPLITS[self.session_id]
        except KeyError as exc:
            raise DatasetError(
                f"session {self.session_id!r} has no fixed train/validation/test split"
            ) from exc


class DatasetError(ValueError):
    pass


def _scalar_text(container: Mapping[str, np.ndarray], key: str) -> str:
    if key not in container:
        raise DatasetError(f"missing scalar field: {key}")
    value = np.asarray(container[key])
    if value.size != 1:
        raise DatasetError(f"{key} must be a scalar")
    return str(value.reshape(-1)[0])


def _label_array(value: np.ndarray, length: int, enum_type: type) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 1:
        array = np.full(length, int(array.reshape(-1)[0]), dtype=np.int16)
    else:
        array = array.reshape(-1).astype(np.int16)
    if len(array) != length:
        raise DatasetError("label length does not match samples")
    allowed = {int(item) for item in enum_type}
    invalid = sorted(set(map(int, np.unique(array))) - allowed)
    if invalid:
        raise DatasetError(f"invalid {enum_type.__name__} labels: {invalid}")
    return array


def load_trial(path: str | Path, *, expected_rate_hz: float = 200.0) -> Trial:
    source = Path(path)
    try:
        handle = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise DatasetError(f"cannot read trial {source}: {exc}") from exc
    required = {"timestamp_ms", "emg", "accel", "gyro", "direction", "gesture", "session_id", "trial_id"}
    missing = sorted(required - set(handle.files))
    if missing:
        raise DatasetError(f"{source} missing fields: {', '.join(missing)}")

    timestamps = np.asarray(handle["timestamp_ms"], dtype=np.float64).reshape(-1)
    emg = np.asarray(handle["emg"], dtype=np.float64)
    accel = np.asarray(handle["accel"], dtype=np.float64)
    gyro = np.asarray(handle["gyro"], dtype=np.float64)
    n = len(timestamps)
    if n < 40 or emg.shape != (n, 8) or accel.shape != (n, 3) or gyro.shape != (n, 3):
        raise DatasetError(f"{source} has invalid sensor shapes")
    if not all(np.isfinite(values).all() for values in (timestamps, emg, accel, gyro)):
        raise DatasetError(f"{source} contains NaN or Inf")
    differences = np.diff(timestamps)
    if np.any(differences <= 0):
        raise DatasetError(f"{source} timestamps are not strictly increasing")
    observed_rate = 1000.0 / float(np.median(differences))
    if not np.isclose(observed_rate, expected_rate_hz, rtol=0.05):
        raise DatasetError(
            f"{source} median sample rate {observed_rate:.2f} Hz differs from {expected_rate_hz:.2f} Hz"
        )

    direction = _label_array(handle["direction"], n, Direction)
    gesture = _label_array(handle["gesture"], n, Gesture)
    if "stable_mask" in handle.files:
        stable_mask = np.asarray(handle["stable_mask"], dtype=bool).reshape(-1)
        if len(stable_mask) != n:
            raise DatasetError("stable_mask length does not match samples")
    else:
        stable_mask = (direction != int(Direction.UNKNOWN)) & (gesture != int(Gesture.UNKNOWN))
    return Trial(
        source,
        _scalar_text(handle, "session_id"),
        _scalar_text(handle, "trial_id"),
        timestamps,
        emg,
        accel,
        gyro,
        direction,
        gesture,
        stable_mask,
        _scalar_text(handle, "session_date") if "session_date" in handle.files else None,
        {
            key: float(np.asarray(handle[key]).reshape(-1)[0])
            for key in EVENT_FIELDS if key in handle.files and np.asarray(handle[key]).size == 1
        },
    )


def discover_trials(root: str | Path) -> list[Trial]:
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise DatasetError(f"no .npz trials found under {root}")
    trials = [load_trial(path) for path in paths]
    validate_no_leakage(trials)
    return trials


def validate_no_leakage(trials: Iterable[Trial]) -> None:
    owners: dict[str, str] = {}
    duplicates: list[str] = []
    for trial in trials:
        if trial.session_id not in SESSION_SPLITS:
            raise DatasetError(
                f"trial {trial.trial_id} has session {trial.session_id!r}; expected 1, 2, 3, or 4"
            )
        split = SESSION_SPLITS[trial.session_id]
        previous = owners.get(trial.trial_id)
        if previous is not None:
            duplicates.append(trial.trial_id)
        else:
            owners[trial.trial_id] = split
    if duplicates:
        raise DatasetError(f"duplicate trial_id (would leak overlapping windows): {sorted(set(duplicates))}")


def trials_for_split(trials: Iterable[Trial], split: str) -> list[Trial]:
    if split not in set(SESSION_SPLITS.values()):
        raise ValueError("split must be train, validation, or test")
    return [trial for trial in trials if trial.split == split]


@dataclass(frozen=True, slots=True)
class FeatureWindows:
    emg: np.ndarray
    imu: np.ndarray
    direction: np.ndarray
    gesture: np.ndarray
    trial_id: np.ndarray
    timestamp_ms: np.ndarray


@dataclass(frozen=True, slots=True)
class RawWindows:
    emg: np.ndarray
    imu: np.ndarray
    direction: np.ndarray
    gesture: np.ndarray
    trial_id: np.ndarray
    timestamp_ms: np.ndarray


def build_feature_windows(
    trials: Iterable[Trial],
    calibrations: Mapping[str, SessionCalibration],
    *,
    window_samples: int = 40,
    hop_samples: int = 8,
    stable_fraction: float = 0.9,
) -> FeatureWindows:
    emg_features: list[np.ndarray] = []
    imu_features: list[np.ndarray] = []
    directions: list[int] = []
    gestures: list[int] = []
    trial_ids: list[str] = []
    timestamps: list[float] = []
    for trial in trials:
        calibration = calibrations.get(trial.session_id)
        if calibration is None:
            raise DatasetError(f"missing calibration for session {trial.session_id}")
        emg_filter = CausalEMGFilter(
            calibration.sample_rate_hz, notch_50hz=calibration.notch_50hz,
        )
        normalized_emg = calibration.transform_emg(emg_filter.process(trial.emg))
        normalized_imu = calibration.transform_imu(trial.accel, trial.gyro)
        for end in range(window_samples, len(trial.timestamp_ms) + 1, hop_samples):
            start = end - window_samples
            if float(trial.stable_mask[start:end].mean()) < stable_fraction:
                continue
            d_values = trial.direction[start:end]
            h_values = trial.gesture[start:end]
            d = int(np.bincount(d_values[d_values >= 0]).argmax()) if np.any(d_values >= 0) else -1
            h = int(np.bincount(h_values[h_values >= 0]).argmax()) if np.any(h_values >= 0) else -1
            if d < 0 or h < 0:
                continue
            emg_features.append(extract_emg_features(normalized_emg[start:end]))
            imu_features.append(extract_imu_features(normalized_imu[start:end]))
            directions.append(d)
            gestures.append(h)
            trial_ids.append(trial.trial_id)
            timestamps.append(float(trial.timestamp_ms[end - 1]))
    if not emg_features:
        raise DatasetError("no stable feature windows were produced")
    return FeatureWindows(
        np.stack(emg_features), np.stack(imu_features),
        np.asarray(directions, dtype=np.int16), np.asarray(gestures, dtype=np.int16),
        np.asarray(trial_ids), np.asarray(timestamps),
    )


def build_raw_windows(
    trials: Iterable[Trial],
    calibrations: Mapping[str, SessionCalibration],
    *,
    window_samples: int = 40,
    hop_samples: int = 8,
    stable_fraction: float = 0.9,
) -> RawWindows:
    emg_windows: list[np.ndarray] = []
    imu_windows: list[np.ndarray] = []
    directions: list[int] = []
    gestures: list[int] = []
    trial_ids: list[str] = []
    timestamps: list[float] = []
    for trial in trials:
        calibration = calibrations.get(trial.session_id)
        if calibration is None:
            raise DatasetError(f"missing calibration for session {trial.session_id}")
        emg_filter = CausalEMGFilter(
            calibration.sample_rate_hz, notch_50hz=calibration.notch_50hz,
        )
        filtered = emg_filter.process(trial.emg)
        normalized_emg = calibration.transform_emg(filtered)
        normalized_imu = calibration.transform_imu(trial.accel, trial.gyro)
        for end in range(window_samples, len(trial.timestamp_ms) + 1, hop_samples):
            start = end - window_samples
            if float(trial.stable_mask[start:end].mean()) < stable_fraction:
                continue
            d_values = trial.direction[start:end]
            h_values = trial.gesture[start:end]
            d = int(np.bincount(d_values[d_values >= 0]).argmax()) if np.any(d_values >= 0) else -1
            h = int(np.bincount(h_values[h_values >= 0]).argmax()) if np.any(h_values >= 0) else -1
            if d < 0 or h < 0:
                continue
            emg_windows.append(normalized_emg[start:end].astype(np.float32))
            imu_windows.append(normalized_imu[start:end].astype(np.float32))
            directions.append(d)
            gestures.append(h)
            trial_ids.append(trial.trial_id)
            timestamps.append(float(trial.timestamp_ms[end - 1]))
    if not emg_windows:
        raise DatasetError("no stable raw windows were produced")
    return RawWindows(
        np.stack(emg_windows), np.stack(imu_windows),
        np.asarray(directions, dtype=np.int64), np.asarray(gestures, dtype=np.int64),
        np.asarray(trial_ids), np.asarray(timestamps),
    )


def dataset_report(trials: Iterable[Trial]) -> dict[str, object]:
    rows = list(trials)
    combinations: dict[str, int] = {}
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    split_combinations: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    session_combination_counts: dict[str, dict[str, int]] = {str(index): {} for index in range(1, 5)}
    onset_offsets: dict[tuple[str, str], list[float]] = {}
    for trial in rows:
        split_counts[trial.split] += 1
        stable = trial.stable_mask
        pairs = set(zip(trial.direction[stable].tolist(), trial.gesture[stable].tolist()))
        for direction, gesture in pairs:
            key = f"{Direction(direction).name}+{Gesture(gesture).name}"
            combinations[key] = combinations.get(key, 0) + 1
            split_combinations[trial.split].add(key)
            per_session = session_combination_counts[trial.session_id]
            per_session[key] = per_session.get(key, 0) + 1
            if direction != int(Direction.NONE) and gesture != int(Gesture.NEUTRAL):
                if "arm_onset_ms" in trial.events and "hand_onset_ms" in trial.events:
                    onset_offsets.setdefault((trial.session_id, key), []).append(
                        trial.events["hand_onset_ms"] - trial.events["arm_onset_ms"]
                    )
    expected = {
        f"{direction.name}+{gesture.name}"
        for direction in Direction if direction != Direction.UNKNOWN
        for gesture in Gesture if gesture != Gesture.UNKNOWN
    }
    dates_by_session = {
        session: sorted({trial.session_date for trial in rows if trial.session_id == session and trial.session_date})
        for session in ("1", "2", "3", "4")
    }
    formal_issues = [
        f"session {session}: {key} has {counts.get(key, 0)} trials; expected at least 3"
        for session, counts in session_combination_counts.items()
        for key in sorted(expected)
        if counts.get(key, 0) < 3
    ]
    missing_dates = [session for session, dates in dates_by_session.items() if not dates]
    if missing_dates:
        formal_issues.append(f"session_date is missing for sessions: {', '.join(missing_dates)}")
    unique_dates = {date for dates in dates_by_session.values() for date in dates}
    if len(unique_dates) < 2:
        formal_issues.append("formal collection must span at least two distinct session_date values")
    compound_expected = {
        f"{direction.name}+{gesture.name}"
        for direction in Direction if direction not in (Direction.UNKNOWN, Direction.NONE)
        for gesture in Gesture if gesture not in (Gesture.UNKNOWN, Gesture.NEUTRAL)
    }
    for session in ("1", "2", "3", "4"):
        for key in sorted(compound_expected):
            offsets = onset_offsets.get((session, key), [])
            missing_targets = [
                target for target in (-200.0, 0.0, 200.0)
                if not any(abs(value - target) <= 80.0 for value in offsets)
            ]
            if missing_targets:
                formal_issues.append(
                    f"session {session}: {key} lacks onset offsets {missing_targets} ms (tolerance 80 ms)"
                )
    return {
        "trial_count": len(rows),
        "split_counts": split_counts,
        "combination_trial_counts": dict(sorted(combinations.items())),
        "missing_combinations": sorted(expected - combinations.keys()),
        "missing_combinations_by_split": {
            split: sorted(expected - present) for split, present in split_combinations.items()
        },
        "session_dates": dates_by_session,
        "formal_collection_issues": formal_issues,
    }


def write_dataset_report(trials: Iterable[Trial], output: str | Path) -> None:
    Path(output).write_text(json.dumps(dataset_report(trials), indent=2, ensure_ascii=False), encoding="utf-8")
