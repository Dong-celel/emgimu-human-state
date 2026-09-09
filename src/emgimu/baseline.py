from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .signal import extract_emg_features, extract_imu_features
from .state import Direction, Gesture


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-4)
    if scaled.ndim == 1:
        scaled = np.column_stack([-scaled, scaled])
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def _decision_logits(model: Any, values: np.ndarray) -> np.ndarray:
    logits = np.asarray(model.decision_function(values))
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    return logits


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Macro F1 over real truth classes; rejected (-1) predictions count as errors."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    scores: list[float] = []
    for label in np.unique(truth):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
    return float(np.mean(scores)) if scores else float("nan")


@dataclass(slots=True)
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> "TemperatureScaler":
        labels = np.asarray(labels)
        class_to_index = {int(value): index for index, value in enumerate(classes)}
        indices = np.array([class_to_index[int(value)] for value in labels], dtype=int)
        best = (float("inf"), 1.0)
        for temperature in np.geomspace(0.25, 4.0, 81):
            probabilities = _softmax(logits, temperature)
            nll = -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-9, 1.0)).mean()
            if nll < best[0]:
                best = (float(nll), float(temperature))
        self.temperature = best[1]
        return self


@dataclass(frozen=True, slots=True)
class HeadPrediction:
    direction: Direction
    gesture: Gesture
    q_direction: float
    q_gesture: float


class BaselinePredictor:
    """Independent RBF-SVM heads with validation-only calibration/rejection."""

    def __init__(self) -> None:
        self.direction_model: Any | None = None
        self.gesture_model: Any | None = None
        self.direction_temperature = TemperatureScaler()
        self.gesture_temperature = TemperatureScaler()
        self.direction_threshold = 0.0
        self.gesture_threshold = 0.0
        self.metadata: dict[str, Any] = {
            "window_ms": 200,
            "hop_ms": 40,
            "sample_rate_hz": 200,
            "test_session_opened": False,
        }

    @staticmethod
    def _new_svm() -> Any:
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required to train the baseline") from exc
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced")),
        ])

    @staticmethod
    def _select_threshold(
        probabilities: np.ndarray,
        predicted_indices: np.ndarray,
        predicted_classes: np.ndarray,
        truth: np.ndarray,
        *,
        min_coverage: float = 0.90,
    ) -> float:
        confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
        best = (-1.0, 0.0)
        for threshold in np.linspace(0.0, 0.95, 96):
            accepted = confidence >= threshold
            coverage = float(accepted.mean())
            if coverage < min_coverage:
                continue
            output = np.where(accepted, predicted_classes, -1)
            score = _macro_f1(truth, output)
            if score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and threshold > best[1]):
                best = (score, float(threshold))
        return best[1]

    def fit(
        self,
        train_emg_features: np.ndarray,
        train_imu_features: np.ndarray,
        train_direction: np.ndarray,
        train_gesture: np.ndarray,
        *,
        validation_emg_features: np.ndarray,
        validation_imu_features: np.ndarray,
        validation_direction: np.ndarray,
        validation_gesture: np.ndarray,
    ) -> "BaselinePredictor":
        self.direction_model = self._new_svm().fit(train_imu_features, train_direction)
        self.gesture_model = self._new_svm().fit(train_emg_features, train_gesture)

        d_logits = _decision_logits(self.direction_model, validation_imu_features)
        h_logits = _decision_logits(self.gesture_model, validation_emg_features)
        d_classes = np.asarray(self.direction_model.classes_)
        h_classes = np.asarray(self.gesture_model.classes_)
        self.direction_temperature.fit(d_logits, validation_direction, d_classes)
        self.gesture_temperature.fit(h_logits, validation_gesture, h_classes)
        d_prob = _softmax(d_logits, self.direction_temperature.temperature)
        h_prob = _softmax(h_logits, self.gesture_temperature.temperature)
        d_index = d_prob.argmax(axis=1)
        h_index = h_prob.argmax(axis=1)
        self.direction_threshold = self._select_threshold(
            d_prob, d_index, d_classes[d_index], validation_direction,
        )
        self.gesture_threshold = self._select_threshold(
            h_prob, h_index, h_classes[h_index], validation_gesture,
        )
        return self

    def _require_fit(self) -> None:
        if self.direction_model is None or self.gesture_model is None:
            raise RuntimeError("baseline predictor has not been fitted")

    def predict_features(self, emg_features: np.ndarray, imu_features: np.ndarray) -> HeadPrediction:
        self._require_fit()
        emg_features = np.asarray(emg_features).reshape(1, -1)
        imu_features = np.asarray(imu_features).reshape(1, -1)
        d_prob = _softmax(
            _decision_logits(self.direction_model, imu_features),
            self.direction_temperature.temperature,
        )[0]
        h_prob = _softmax(
            _decision_logits(self.gesture_model, emg_features),
            self.gesture_temperature.temperature,
        )[0]
        d_index, h_index = int(d_prob.argmax()), int(h_prob.argmax())
        qd, qh = float(d_prob[d_index]), float(h_prob[h_index])
        d_class = int(self.direction_model.classes_[d_index])
        h_class = int(self.gesture_model.classes_[h_index])
        direction = Direction(d_class) if qd >= self.direction_threshold else Direction.UNKNOWN
        gesture = Gesture(h_class) if qh >= self.gesture_threshold else Gesture.UNKNOWN
        return HeadPrediction(direction, gesture, qd, qh)

    def predict(self, normalized_emg: np.ndarray, normalized_imu: np.ndarray) -> HeadPrediction:
        return self.predict_features(
            extract_emg_features(normalized_emg),
            extract_imu_features(normalized_imu),
        )

    def save(self, path: str | Path) -> None:
        self._require_fit()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "BaselinePredictor":
        with Path(path).open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, cls):
            raise TypeError("artifact is not a BaselinePredictor")
        return value


def fit_lda_sanity_baselines(
    emg_features: np.ndarray,
    imu_features: np.ndarray,
    gestures: Iterable[int],
    directions: Iterable[int],
) -> tuple[Any, Any]:
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required to train LDA baselines") from exc
    gesture_model = Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ]).fit(emg_features, np.asarray(list(gestures)))
    direction_model = Pipeline([
        ("scale", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ]).fit(imu_features, np.asarray(list(directions)))
    return gesture_model, direction_model
