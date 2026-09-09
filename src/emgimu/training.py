from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import BaselinePredictor, TemperatureScaler, _softmax
from .data import RawWindows
from .neural import (
    DualBranchCausalNet,
    DualBranchConfig,
    NeuralPredictor,
    augment_training_batch,
    dual_branch_loss,
    torch,
)


@dataclass(frozen=True, slots=True)
class NeuralTrainingResult:
    predictor: NeuralPredictor
    best_epoch: int
    validation_loss: float
    history: tuple[dict[str, float], ...]


def _class_weights(labels: np.ndarray, class_count: int, device: str) -> "torch.Tensor":
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _batched_indices(length: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator):
    indices = np.arange(length)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start:start + batch_size]


def train_dual_branch(
    train: RawWindows,
    validation: RawWindows,
    *,
    config: DualBranchConfig | None = None,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> NeuralTrainingResult:
    if torch is None:
        raise RuntimeError("install the neural extra before training")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = config or DualBranchConfig()
    model = DualBranchCausalNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    d_weight = _class_weights(train.direction, cfg.direction_classes, device)
    h_weight = _class_weights(train.gesture, cfg.gesture_classes, device)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(int(max_epochs)):
        model.train()
        train_total = 0.0
        train_count = 0
        for indices in _batched_indices(len(train.emg), batch_size, shuffle=True, rng=rng):
            emg = torch.as_tensor(train.emg[indices], dtype=torch.float32, device=device)
            imu = torch.as_tensor(train.imu[indices], dtype=torch.float32, device=device)
            direction = torch.as_tensor(train.direction[indices], dtype=torch.long, device=device)
            gesture = torch.as_tensor(train.gesture[indices], dtype=torch.long, device=device)
            emg, imu = augment_training_batch(emg, imu)
            optimizer.zero_grad(set_to_none=True)
            loss = dual_branch_loss(model(emg, imu), direction, gesture,
                                    direction_weight=d_weight, gesture_weight=h_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
            train_count += len(indices)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.inference_mode():
            for indices in _batched_indices(len(validation.emg), batch_size, shuffle=False, rng=rng):
                emg = torch.as_tensor(validation.emg[indices], dtype=torch.float32, device=device)
                imu = torch.as_tensor(validation.imu[indices], dtype=torch.float32, device=device)
                direction = torch.as_tensor(validation.direction[indices], dtype=torch.long, device=device)
                gesture = torch.as_tensor(validation.gesture[indices], dtype=torch.long, device=device)
                loss = dual_branch_loss(model(emg, imu), direction, gesture,
                                        direction_weight=d_weight, gesture_weight=h_weight)
                validation_total += float(loss) * len(indices)
                validation_count += len(indices)
        train_loss = train_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_count, 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("neural training did not produce a checkpoint")
    model.load_state_dict(best_state)
    d_logits, h_logits = _collect_logits(model, validation, batch_size=batch_size, device=device)
    d_classes = np.arange(cfg.direction_classes)
    h_classes = np.arange(cfg.gesture_classes)
    d_temp = TemperatureScaler().fit(d_logits, validation.direction, d_classes).temperature
    h_temp = TemperatureScaler().fit(h_logits, validation.gesture, h_classes).temperature
    d_prob = _softmax(d_logits, d_temp)
    h_prob = _softmax(h_logits, h_temp)
    d_index = d_prob.argmax(axis=1)
    h_index = h_prob.argmax(axis=1)
    d_threshold = BaselinePredictor._select_threshold(
        d_prob, d_index, d_index, validation.direction,
    )
    h_threshold = BaselinePredictor._select_threshold(
        h_prob, h_index, h_index, validation.gesture,
    )
    predictor = NeuralPredictor(
        model,
        direction_temperature=d_temp,
        gesture_temperature=h_temp,
        direction_threshold=d_threshold,
        gesture_threshold=h_threshold,
        device=device,
    )
    return NeuralTrainingResult(predictor, best_epoch, best_loss, tuple(history))


def _collect_logits(
    model: "DualBranchCausalNet",
    windows: RawWindows,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    direction: list[np.ndarray] = []
    gesture: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for indices in _batched_indices(len(windows.emg), batch_size, shuffle=False, rng=rng):
            output = model(
                torch.as_tensor(windows.emg[indices], dtype=torch.float32, device=device),
                torch.as_tensor(windows.imu[indices], dtype=torch.float32, device=device),
            )
            direction.append(output["direction"].cpu().numpy())
            gesture.append(output["gesture"].cpu().numpy())
    return np.concatenate(direction), np.concatenate(gesture)


def save_neural_artifact(result: NeuralTrainingResult, path: str | Path) -> None:
    if torch is None:
        raise RuntimeError("install the neural extra before saving")
    predictor = result.predictor
    payload = {
        "format_version": 1,
        "config": predictor.model.config.to_dict(),
        "state_dict": predictor.model.state_dict(),
        "direction_temperature": predictor.direction_temperature,
        "gesture_temperature": predictor.gesture_temperature,
        "direction_threshold": predictor.direction_threshold,
        "gesture_threshold": predictor.gesture_threshold,
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
        "history": list(result.history),
        "test_session_opened": False,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_neural_artifact(path: str | Path, *, device: str = "cpu") -> NeuralPredictor:
    if torch is None:
        raise RuntimeError("install the neural extra before loading")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported neural artifact format")
    model = DualBranchCausalNet(DualBranchConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return NeuralPredictor(
        model,
        direction_temperature=payload["direction_temperature"],
        gesture_temperature=payload["gesture_temperature"],
        direction_threshold=payload["direction_threshold"],
        gesture_threshold=payload["gesture_threshold"],
        device=device,
    )
