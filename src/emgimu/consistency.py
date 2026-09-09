from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .state import Consistency, Direction, Gesture, Phase


ConsistencyKey = tuple[int, int, int, int]


def state_key(direction: Direction, gesture: Gesture, arm_phase: Phase, hand_phase: Phase) -> ConsistencyKey:
    return int(direction), int(gesture), int(arm_phase), int(hand_phase)


@dataclass(slots=True)
class _ConditionalStats:
    center: np.ndarray
    inverse_covariance: np.ndarray


class ConditionalConsistencyModel:
    """Shadow-only conditional anomaly detector for [A, M, onset lag]."""

    def __init__(self, *, min_samples: int = 12, atypical_score: float = 0.8) -> None:
        self.min_samples = int(min_samples)
        self.atypical_score = float(atypical_score)
        self.stats: dict[ConsistencyKey, _ConditionalStats] = {}

    def fit(self, records: Iterable[tuple[ConsistencyKey, float, float, float]]) -> "ConditionalConsistencyModel":
        grouped: dict[ConsistencyKey, list[list[float]]] = {}
        for key, activation, motion, onset_lag_ms in records:
            values = [float(activation), float(motion), float(onset_lag_ms) / 300.0]
            if np.isfinite(values).all():
                grouped.setdefault(tuple(map(int, key)), []).append(values)
        self.stats.clear()
        for key, rows in grouped.items():
            if len(rows) < self.min_samples:
                continue
            x = np.asarray(rows, dtype=np.float64)
            center = np.median(x, axis=0)
            covariance = np.cov(x, rowvar=False)
            covariance = np.atleast_2d(covariance) + np.eye(3) * 1e-3
            self.stats[key] = _ConditionalStats(center, np.linalg.pinv(covariance))
        return self

    def predict(
        self,
        key: ConsistencyKey,
        activation: float,
        motion: float,
        onset_lag_ms: float | None,
    ) -> tuple[Consistency, float | None]:
        stats = self.stats.get(tuple(map(int, key)))
        if stats is None or onset_lag_ms is None:
            return Consistency.UNKNOWN, None
        value = np.array([activation, motion, onset_lag_ms / 300.0], dtype=np.float64)
        delta = value - stats.center
        distance_sq = max(float(delta @ stats.inverse_covariance @ delta), 0.0)
        score = float(1.0 - np.exp(-0.5 * distance_sq))
        status = Consistency.ATYPICAL if score >= self.atypical_score else Consistency.NORMAL
        return status, score

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "min_samples": self.min_samples,
            "atypical_score": self.atypical_score,
            "stats": {
                ",".join(map(str, key)): {
                    "center": value.center.tolist(),
                    "inverse_covariance": value.inverse_covariance.tolist(),
                }
                for key, value in self.stats.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConditionalConsistencyModel":
        if payload.get("format_version") != 1:
            raise ValueError("unsupported consistency model format")
        model = cls(
            min_samples=int(payload["min_samples"]),
            atypical_score=float(payload["atypical_score"]),
        )
        for raw_key, raw_value in dict(payload["stats"]).items():
            key = tuple(map(int, str(raw_key).split(",")))
            if len(key) != 4:
                raise ValueError("invalid consistency key")
            value = dict(raw_value)
            model.stats[key] = _ConditionalStats(
                np.asarray(value["center"], dtype=np.float64).reshape(3),
                np.asarray(value["inverse_covariance"], dtype=np.float64).reshape(3, 3),
            )
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalConsistencyModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
