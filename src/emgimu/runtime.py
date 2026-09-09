from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from .baseline import HeadPrediction
from .calibration import SessionCalibration
from .consistency import ConditionalConsistencyModel, state_key
from .phases import PhaseTracker
from .signal import (
    CausalEMGFilter,
    activation_intensity,
    estimate_signal_quality,
    motion_intensity,
)
from .state import (
    Confidence,
    Consistency,
    Direction,
    Gesture,
    HumanState,
    Phase,
    PhasePair,
    QualityFlag,
    SignalQuality,
)


class WindowPredictor(Protocol):
    def predict(self, normalized_emg: np.ndarray, normalized_imu: np.ndarray) -> HeadPrediction: ...


class HumanStateEstimator:
    """Causal 200 Hz input / 25 Hz output human-state estimator."""

    def __init__(
        self,
        predictor: WindowPredictor,
        calibration: SessionCalibration,
        *,
        window_samples: int = 40,
        hop_samples: int = 8,
        quality_threshold: float = 0.60,
        notch_50hz: bool | None = None,
        consistency_model: ConditionalConsistencyModel | None = None,
    ) -> None:
        if window_samples < 3 or hop_samples < 1 or hop_samples > window_samples:
            raise ValueError("invalid window or hop length")
        self.predictor = predictor
        self.calibration = calibration
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.quality_threshold = float(quality_threshold)
        self.consistency_model = consistency_model
        use_notch = calibration.notch_50hz if notch_50hz is None else bool(notch_50hz)
        self.filter = CausalEMGFilter(calibration.sample_rate_hz, notch_50hz=use_notch)
        self.arm_phase = PhaseTracker(Direction.NONE, Direction.UNKNOWN)
        self.hand_phase = PhaseTracker(Gesture.NEUTRAL, Gesture.UNKNOWN)
        self.raw_emg: deque[np.ndarray] = deque(maxlen=window_samples)
        self.filtered_emg: deque[np.ndarray] = deque(maxlen=window_samples)
        self.raw_imu: deque[np.ndarray] = deque(maxlen=window_samples)
        self.timestamps: deque[int] = deque(maxlen=window_samples)
        self.samples_since_output = 0
        self.last_timestamp_ms = -1
        self.dropped_samples = False
        self.last_arm_onset_ms: int | None = None
        self.last_hand_onset_ms: int | None = None
        self.current_onset_lag_ms: float | None = None

    def reset(self) -> None:
        self.filter.reset()
        self.arm_phase.reset()
        self.hand_phase.reset()
        self.raw_emg.clear()
        self.filtered_emg.clear()
        self.raw_imu.clear()
        self.timestamps.clear()
        self.samples_since_output = 0
        self.last_timestamp_ms = -1
        self.dropped_samples = False
        self.last_arm_onset_ms = None
        self.last_hand_onset_ms = None
        self.current_onset_lag_ms = None

    def push_sample(
        self,
        timestamp_ms: int,
        emg: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        *,
        interpolated_imu: bool = False,
    ) -> HumanState | None:
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms <= self.last_timestamp_ms:
            return None
        expected_period = 1000.0 / self.calibration.sample_rate_hz
        if self.last_timestamp_ms >= 0 and timestamp_ms - self.last_timestamp_ms > expected_period * 2.5:
            self.dropped_samples = True
        self.last_timestamp_ms = timestamp_ms

        emg_sample = np.asarray(emg, dtype=np.float64).reshape(8)
        accel_sample = np.asarray(accel, dtype=np.float64).reshape(3)
        gyro_sample = np.asarray(gyro, dtype=np.float64).reshape(3)
        raw_imu_sample = np.concatenate([accel_sample, gyro_sample])
        safe_emg = np.nan_to_num(emg_sample)
        filtered = self.filter.process(safe_emg[None, :])[0]
        self.raw_emg.append(emg_sample)
        self.filtered_emg.append(filtered)
        self.raw_imu.append(raw_imu_sample)
        self.timestamps.append(timestamp_ms)
        self.samples_since_output += 1
        if len(self.timestamps) < self.window_samples or self.samples_since_output < self.hop_samples:
            return None
        self.samples_since_output = 0
        return self._estimate(interpolated_imu=interpolated_imu)

    def push_batch(
        self,
        timestamp_ms: np.ndarray,
        emg: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> list[HumanState]:
        timestamps = np.asarray(timestamp_ms).reshape(-1)
        emg = np.asarray(emg)
        accel = np.asarray(accel)
        gyro = np.asarray(gyro)
        n = len(timestamps)
        if emg.shape != (n, 8) or accel.shape != (n, 3) or gyro.shape != (n, 3):
            raise ValueError("batch sensor shapes do not match timestamps")
        states: list[HumanState] = []
        for index in range(n):
            state = self.push_sample(timestamps[index], emg[index], accel[index], gyro[index])
            if state is not None:
                states.append(state)
        return states

    def _estimate(self, *, interpolated_imu: bool) -> HumanState:
        timestamp_ms = self.timestamps[-1]
        raw_emg = np.stack(self.raw_emg)
        filtered_emg = np.stack(self.filtered_emg)
        raw_imu = np.stack(self.raw_imu)
        normalized_emg = self.calibration.transform_emg(filtered_emg)
        normalized_imu = self.calibration.transform_imu(raw_imu[:, :3], raw_imu[:, 3:])
        emg_quality, imu_quality = estimate_signal_quality(
            raw_emg, raw_imu, expected_samples=self.window_samples,
        )
        flags = QualityFlag.TIMESTAMP_VALID | QualityFlag.CALIBRATED
        if emg_quality >= self.quality_threshold:
            flags |= QualityFlag.EMG_VALID
        if imu_quality >= self.quality_threshold:
            flags |= QualityFlag.IMU_VALID
        if interpolated_imu:
            flags |= QualityFlag.INTERPOLATED_IMU
        if self.dropped_samples:
            flags |= QualityFlag.DROPPED_SAMPLES
            self.dropped_samples = False
        quality = SignalQuality(emg_quality, imu_quality, flags)

        prediction = self.predictor.predict(normalized_emg, normalized_imu)
        raw_direction = prediction.direction if imu_quality >= self.quality_threshold else Direction.UNKNOWN
        raw_gesture = prediction.gesture if emg_quality >= self.quality_threshold else Gesture.UNKNOWN
        arm = self.arm_phase.update(raw_direction, prediction.q_direction, valid=raw_direction != Direction.UNKNOWN)
        hand = self.hand_phase.update(raw_gesture, prediction.q_gesture, valid=raw_gesture != Gesture.UNKNOWN)
        direction = Direction(arm.stable_label)
        gesture = Gesture(hand.stable_label)

        if arm.phase == Phase.ONSET:
            self.last_arm_onset_ms = timestamp_ms
        if hand.phase == Phase.ONSET:
            self.last_hand_onset_ms = timestamp_ms
        onset_lag: float | None = None
        if self.last_arm_onset_ms is not None and self.last_hand_onset_ms is not None:
            candidate_lag = float(self.last_arm_onset_ms - self.last_hand_onset_ms)
            if abs(candidate_lag) <= 600.0:
                onset_lag = candidate_lag
        self.current_onset_lag_ms = onset_lag

        activation = activation_intensity(normalized_emg)
        motion = motion_intensity(normalized_imu)
        consistency = Consistency.UNKNOWN
        consistency_score = None
        if self.consistency_model is not None:
            consistency, consistency_score = self.consistency_model.predict(
                state_key(direction, gesture, arm.phase, hand.phase),
                activation, motion, onset_lag,
            )
        arm_q = prediction.q_direction * imu_quality if arm.phase != Phase.UNKNOWN else 0.0
        hand_q = prediction.q_gesture * emg_quality if hand.phase != Phase.UNKNOWN else 0.0
        return HumanState(
            timestamp_ms=timestamp_ms,
            direction=direction,
            gesture=gesture,
            activation=activation,
            phase=PhasePair(arm.phase, hand.phase),
            motion_intensity=motion,
            consistency=consistency,
            confidence=Confidence(
                prediction.q_direction,
                prediction.q_gesture,
                arm_q,
                hand_q,
            ),
            signal_quality=quality,
            consistency_score=consistency_score,
        )
