from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    direction_macro_f1: float
    gesture_macro_f1: float
    joint_accuracy: float
    direction_coverage: float
    gesture_coverage: float
    direction_ece: float
    gesture_ece: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    direction_labels: tuple[int, ...]
    gesture_labels: tuple[int, ...]
    direction_confusion_matrix: tuple[tuple[int, ...], ...]
    gesture_confusion_matrix: tuple[tuple[int, ...], ...]

    @property
    def meets_v1_target(self) -> bool:
        latency_ok = self.p95_latency_ms is not None and self.p95_latency_ms <= 300.0
        return (
            self.direction_macro_f1 >= 0.85
            and self.gesture_macro_f1 >= 0.85
            and self.joint_accuracy >= 0.75
            and latency_ok
        )


def expected_calibration_error(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    confidence = np.asarray(confidence, dtype=np.float64)
    total = len(truth)
    if total == 0:
        return float("nan")
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        accuracy = np.mean(truth[mask] == predicted[mask])
        result += float(mask.mean()) * abs(float(accuracy) - float(confidence[mask].mean()))
    return result


def evaluate_predictions(
    direction_truth: np.ndarray,
    gesture_truth: np.ndarray,
    direction_predicted: np.ndarray,
    gesture_predicted: np.ndarray,
    q_direction: np.ndarray,
    q_gesture: np.ndarray,
    *,
    latency_ms: np.ndarray | None = None,
) -> EvaluationResult:
    d_true = np.asarray(direction_truth)
    h_true = np.asarray(gesture_truth)
    d_pred = np.asarray(direction_predicted)
    h_pred = np.asarray(gesture_predicted)
    if not (len(d_true) == len(h_true) == len(d_pred) == len(h_pred)):
        raise ValueError("prediction arrays must have equal lengths")
    p50 = None
    p95 = None
    if latency_ms is not None and len(latency_ms):
        latency_values = np.asarray(latency_ms, dtype=np.float64)
        p50 = float(np.percentile(latency_values, 50))
        p95 = float(np.percentile(latency_values, 95))
    d_labels = tuple(map(int, sorted(set(d_true.tolist()) | {-1})))
    h_labels = tuple(map(int, sorted(set(h_true.tolist()) | {-1})))
    def confusion(truth: np.ndarray, predicted: np.ndarray, labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        positions = {label: index for index, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for actual, estimate in zip(truth, predicted):
            if int(actual) in positions and int(estimate) in positions:
                matrix[positions[int(actual)], positions[int(estimate)]] += 1
        return tuple(tuple(map(int, row)) for row in matrix)
    def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
        values = []
        for label in np.unique(truth):
            tp = np.sum((truth == label) & (predicted == label))
            fp = np.sum((truth != label) & (predicted == label))
            fn = np.sum((truth == label) & (predicted != label))
            denominator = 2 * tp + fp + fn
            values.append(0.0 if denominator == 0 else float(2 * tp / denominator))
        return float(np.mean(values)) if values else float("nan")
    return EvaluationResult(
        direction_macro_f1=macro_f1(d_true, d_pred),
        gesture_macro_f1=macro_f1(h_true, h_pred),
        joint_accuracy=float(np.mean((d_true == d_pred) & (h_true == h_pred))),
        direction_coverage=float(np.mean(d_pred >= 0)),
        gesture_coverage=float(np.mean(h_pred >= 0)),
        direction_ece=expected_calibration_error(d_true, d_pred, q_direction),
        gesture_ece=expected_calibration_error(h_true, h_pred, q_gesture),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        direction_labels=d_labels,
        gesture_labels=h_labels,
        direction_confusion_matrix=confusion(d_true, d_pred, d_labels),
        gesture_confusion_matrix=confusion(h_true, h_pred, h_labels),
    )


def first_stable_latency(
    onset_ms: float,
    state_timestamps_ms: np.ndarray,
    predicted_labels: np.ndarray,
    target_label: int,
    *,
    maximum_ms: float = 1000.0,
) -> float | None:
    timestamps = np.asarray(state_timestamps_ms, dtype=np.float64)
    labels = np.asarray(predicted_labels)
    mask = (timestamps >= onset_ms) & (timestamps <= onset_ms + maximum_ms) & (labels == target_label)
    matches = np.flatnonzero(mask)
    return None if not len(matches) else float(timestamps[matches[0]] - onset_ms)
