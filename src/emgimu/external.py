from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .signal import polyphase_resample
from .state import Direction, Gesture


class ChannelPolicy(str, Enum):
    CIRCULAR_8 = "circular_8"
    SPLIT_MYOS = "split_myos"
    SOURCE_ADAPTER_ONLY = "source_adapter_only"
    ARCHITECTURE_REFERENCE_ONLY = "architecture_reference_only"


@dataclass(frozen=True, slots=True)
class ExternalDatasetSpec:
    name: str
    emg_channels: int
    emg_rate_hz: float
    has_imu: bool
    channel_policy: ChannelPolicy
    license_id: str
    commercial_allowed: bool
    role: str
    source_url: str

    def assert_allowed(self, *, commercial_use: bool) -> None:
        if commercial_use and not self.commercial_allowed:
            raise PermissionError(f"{self.name} is not allowed in commercial mode ({self.license_id})")


DATASETS: dict[str, ExternalDatasetSpec] = {
    "epn100_myo": ExternalDatasetSpec(
        "EMG-IMU-EPN-100+ (Myo)", 8, 200, True, ChannelPolicy.CIRCULAR_8,
        "verify-at-download", False, "pretrain D and H separately",
        "https://laboratorio-ia.epn.edu.ec/en/resources/dataset/emg-imu-epn-100",
    ),
    "epn612": ExternalDatasetSpec(
        "EMG-EPN-612", 8, 200, False, ChannelPolicy.CIRCULAR_8,
        "verify-at-download", False, "pretrain H",
        "https://www.libemg.com/",
    ),
    "electrode_shift": ExternalDatasetSpec(
        "Electrode Shift", 8, 200, False, ChannelPolicy.CIRCULAR_8,
        "verify-at-download", False, "rotation robustness",
        "https://www.libemg.com/",
    ),
    "ninapro_db5": ExternalDatasetSpec(
        "NinaPro DB5", 16, 200, True, ChannelPolicy.SPLIT_MYOS,
        "NinaPro terms", False, "separate-band transfer experiment",
        "https://ninapro.hevs.ch/",
    ),
    "meta_gni": ExternalDatasetSpec(
        "Meta Generic Neuromotor Interface", 16, 2000, True,
        ChannelPolicy.ARCHITECTURE_REFERENCE_ONLY, "CC-BY-NC-4.0", False,
        "architecture and onset-decoding reference",
        "https://github.com/facebookresearch/generic-neuromotor-interface",
    ),
    "3dc": ExternalDatasetSpec(
        "3DC", 10, 1000, False, ChannelPolicy.SOURCE_ADAPTER_ONLY,
        "verify-at-download", False, "feature benchmark",
        "https://github.com/LibEMG/3DCDataset",
    ),
    "grabmyo": ExternalDatasetSpec(
        "GRABMyo", 16, 2048, False, ChannelPolicy.SOURCE_ADAPTER_ONLY,
        "CC-BY-4.0", True, "cross-session feature benchmark",
        "https://physionet.org/content/grabmyo/",
    ),
}


GESTURE_LABEL_MAP: dict[str, Gesture] = {
    "relax": Gesture.NEUTRAL,
    "rest": Gesture.NEUTRAL,
    "no movement": Gesture.NEUTRAL,
    "hand close": Gesture.FIST,
    "close": Gesture.FIST,
    "fist": Gesture.FIST,
    "hand open": Gesture.OPEN,
    "open": Gesture.OPEN,
    "pinch": Gesture.PINCH,
    "pinch grip": Gesture.PINCH,
}

DIRECTION_LABEL_MAP: dict[str, Direction] = {
    "up": Direction.UP,
    "down": Direction.DOWN,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
    "forward": Direction.FORWARD,
    "backward": Direction.BACKWARD,
    "back": Direction.BACKWARD,
    "relax": Direction.NONE,
    "rest": Direction.NONE,
}


def adapt_external_emg(
    name: str,
    emg: np.ndarray,
    *,
    commercial_use: bool = False,
    target_rate_hz: float = 200.0,
) -> list[np.ndarray]:
    """Apply only defensible rate/channel adaptations; never fake electrodes."""
    spec = DATASETS[name]
    spec.assert_allowed(commercial_use=commercial_use)
    values = np.asarray(emg, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != spec.emg_channels:
        raise ValueError(f"{name} expected [samples,{spec.emg_channels}] EMG")
    if spec.emg_rate_hz < target_rate_hz:
        raise ValueError("lower-rate EMG is not upsampled for raw-model pretraining")
    if spec.emg_rate_hz != target_rate_hz:
        values = polyphase_resample(values, spec.emg_rate_hz, target_rate_hz)
    if spec.channel_policy == ChannelPolicy.CIRCULAR_8:
        return [values]
    if spec.channel_policy == ChannelPolicy.SPLIT_MYOS:
        return [values[:, :8], values[:, 8:16]]
    raise ValueError(f"{name} may not enter the shared raw 8-channel model")


def canonical_label(
    label: str,
    mapping: Mapping[str, Direction | Gesture],
) -> Direction | Gesture:
    key = label.strip().lower()
    if key not in mapping:
        raise KeyError(f"label {label!r} has no approved canonical mapping")
    return mapping[key]

