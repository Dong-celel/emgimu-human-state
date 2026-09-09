from __future__ import annotations

import socket
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from typing import Callable

import numpy as np

from .osc import OscPublisher, decode_message
from .runtime import HumanStateEstimator
from .state import HumanState


RAW_OSC_ADDRESS = "/emgimu/raw"


@dataclass(frozen=True, slots=True)
class RawSample:
    timestamp_ms: int
    emg: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray


def parse_raw_message(packet: bytes, *, address: str = RAW_OSC_ADDRESS) -> RawSample | None:
    actual_address, args = decode_message(packet)
    if actual_address != address:
        return None
    if len(args) != 15:
        raise ValueError(f"{address} expects timestamp + 8 EMG + 3 accel + 3 gyro values")
    return RawSample(
        int(args[0]),
        np.asarray(args[1:9], dtype=np.float64),
        np.asarray(args[9:12], dtype=np.float64),
        np.asarray(args[12:15], dtype=np.float64),
    )


class RawOscServer:
    def __init__(
        self,
        callback: Callable[[RawSample], None],
        host: str = "127.0.0.1",
        port: int = 9100,
    ) -> None:
        self.callback = callback
        self.host = host
        self.port = int(port)
        self.last_error: Exception | None = None

    def run_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            self.port = int(sock.getsockname()[1])
            while True:
                packet, _ = sock.recvfrom(8192)
                try:
                    sample = parse_raw_message(packet)
                    if sample is not None:
                        self.callback(sample)
                except (ValueError, TypeError) as exc:
                    self.last_error = exc


class LiveClassifierService:
    def __init__(
        self,
        estimator: HumanStateEstimator,
        publisher: OscPublisher,
        logger: "StateCsvLogger | None" = None,
    ) -> None:
        self.estimator = estimator
        self.publisher = publisher
        self.logger = logger

    def accept(self, sample: RawSample) -> HumanState | None:
        state = self.estimator.push_sample(
            sample.timestamp_ms, sample.emg, sample.accel, sample.gyro,
        )
        if state is not None:
            self.publisher.publish(state)
            if self.logger is not None:
                self.logger.write(state, self.estimator.current_onset_lag_ms)
        return state


class StateCsvLogger:
    """Append runtime observations used to fit the shadow consistency model."""

    FIELDS = (
        "timestamp_ms", "direction", "gesture", "arm_phase", "hand_phase",
        "activation", "motion", "onset_lag_ms", "q_direction", "q_gesture",
        "emg_quality", "imu_quality",
    )

    def __init__(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        existed = output.exists() and output.stat().st_size > 0
        self._handle: TextIO = output.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        if not existed:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, state: HumanState, onset_lag_ms: float | None) -> None:
        self._writer.writerow({
            "timestamp_ms": state.timestamp_ms,
            "direction": int(state.direction),
            "gesture": int(state.gesture),
            "arm_phase": int(state.phase.arm),
            "hand_phase": int(state.phase.hand),
            "activation": state.activation,
            "motion": state.motion_intensity,
            "onset_lag_ms": "" if onset_lag_ms is None else onset_lag_ms,
            "q_direction": state.confidence.direction,
            "q_gesture": state.confidence.gesture,
            "emg_quality": state.signal_quality.emg,
            "imu_quality": state.signal_quality.imu,
        })
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "StateCsvLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
