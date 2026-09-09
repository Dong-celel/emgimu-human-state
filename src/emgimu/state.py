from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum, IntFlag
from typing import Any


class Direction(IntEnum):
    UNKNOWN = -1
    NONE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    UP = 5
    DOWN = 6


class Gesture(IntEnum):
    UNKNOWN = -1
    NEUTRAL = 0
    PINCH = 1
    FIST = 2
    OPEN = 3


class Phase(IntEnum):
    UNKNOWN = -1
    IDLE = 0
    ONSET = 1
    ACTIVE = 2
    HOLD = 3
    RELEASE = 4
    TRANSITION = 5


class Consistency(IntEnum):
    UNKNOWN = -1
    NORMAL = 0
    ATYPICAL = 1


class QualityFlag(IntFlag):
    NONE = 0
    TIMESTAMP_VALID = 1 << 0
    EMG_VALID = 1 << 1
    IMU_VALID = 1 << 2
    CALIBRATED = 1 << 3
    INTERPOLATED_IMU = 1 << 4
    DROPPED_SAMPLES = 1 << 5


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class PhasePair:
    arm: Phase
    hand: Phase


@dataclass(frozen=True, slots=True)
class Confidence:
    direction: float
    gesture: float
    arm_phase: float
    hand_phase: float

    def __post_init__(self) -> None:
        for name in ("direction", "gesture", "arm_phase", "hand_phase"):
            object.__setattr__(self, name, clamp01(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SignalQuality:
    emg: float
    imu: float
    flags: QualityFlag = QualityFlag.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "emg", clamp01(self.emg))
        object.__setattr__(self, "imu", clamp01(self.imu))
        object.__setattr__(self, "flags", QualityFlag(self.flags))

    @property
    def valid(self) -> bool:
        required = QualityFlag.TIMESTAMP_VALID | QualityFlag.EMG_VALID | QualityFlag.IMU_VALID
        return (self.flags & required) == required


@dataclass(frozen=True, slots=True)
class HumanState:
    timestamp_ms: int
    direction: Direction
    gesture: Gesture
    activation: float
    phase: PhasePair
    motion_intensity: float
    consistency: Consistency
    confidence: Confidence
    signal_quality: SignalQuality
    consistency_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "gesture", Gesture(self.gesture))
        object.__setattr__(self, "activation", clamp01(self.activation))
        object.__setattr__(self, "motion_intensity", clamp01(self.motion_intensity))
        object.__setattr__(self, "consistency", Consistency(self.consistency))
        if self.consistency_score is not None:
            object.__setattr__(self, "consistency_score", clamp01(self.consistency_score))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction"] = self.direction.name.lower()
        result["gesture"] = self.gesture.name.lower()
        result["phase"] = {
            "arm": self.phase.arm.name.lower(),
            "hand": self.phase.hand.name.lower(),
        }
        result["consistency"] = self.consistency.name.lower()
        result["signal_quality"]["flags"] = int(self.signal_quality.flags)
        return result

    def osc_v2_args(self) -> tuple[int | float, ...]:
        score = -1.0 if self.consistency_score is None else self.consistency_score
        return (
            self.timestamp_ms,
            int(self.direction),
            int(self.gesture),
            int(self.phase.arm),
            int(self.phase.hand),
            self.activation,
            self.motion_intensity,
            int(self.consistency),
            score,
            self.confidence.direction,
            self.confidence.gesture,
            self.confidence.arm_phase,
            self.confidence.hand_phase,
            self.signal_quality.emg,
            self.signal_quality.imu,
            int(self.signal_quality.flags),
        )

    def legacy_args(self) -> tuple[int | float, ...]:
        # The legacy music frontend only understands six directions. Unknown and
        # stationary states both map to NONE; hand posture remains available in v2.
        gesture_id = max(0, int(self.direction))
        return (
            self.timestamp_ms,
            gesture_id,
            self.confidence.direction,
            self.motion_intensity,
            self.activation,
        )
