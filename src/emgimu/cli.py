from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baseline import BaselinePredictor, fit_lda_sanity_baselines, _macro_f1
from .calibration import SessionCalibration, fit_session_calibration
from .data import (
    build_feature_windows,
    build_raw_windows,
    dataset_report,
    discover_trials,
    trials_for_split,
)
from .metrics import evaluate_predictions
from .metrics import first_stable_latency
from .training import load_neural_artifact, save_neural_artifact, train_dual_branch
from .simulate import create_smoke_dataset
from .runtime import HumanStateEstimator
from .consistency import ConditionalConsistencyModel
from .osc import OscPublisher
from .service import LiveClassifierService, RawOscServer, StateCsvLogger
from .state import Direction, Gesture


def _calibration_directory(root: str | Path) -> Path:
    return Path(root) / "calibration"


def load_calibrations(root: str | Path) -> dict[str, SessionCalibration]:
    result: dict[str, SessionCalibration] = {}
    for session_id in ("1", "2", "3", "4"):
        path = _calibration_directory(root) / f"session_{session_id}.json"
        if path.exists():
            result[session_id] = SessionCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return result


def cmd_validate(args: argparse.Namespace) -> int:
    report = dataset_report(discover_trials(args.dataset))
    calibrations = load_calibrations(args.dataset)
    report["missing_calibrations"] = [
        session for session in ("1", "2", "3", "4") if session not in calibrations
    ]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    has_missing_split = any(report["missing_combinations_by_split"].values())
    formal_failed = bool(args.formal and report["formal_collection_issues"])
    return 1 if report["missing_combinations"] or has_missing_split or report["missing_calibrations"] or formal_failed else 0


def cmd_make_smoke(args: argparse.Namespace) -> int:
    output = create_smoke_dataset(args.output, repetitions=args.repetitions)
    print(str(output))
    return 0


def cmd_fit_calibration(args: argparse.Namespace) -> int:
    source = np.load(args.input, allow_pickle=False)
    directions = {}
    for name in ("forward", "backward", "left", "right", "up", "down"):
        key = f"direction_{name}"
        if key in source.files:
            from .state import Direction
            directions[Direction[name.upper()]] = np.asarray(source[key]).reshape(3)
    reference = source["reference_emg_profile"] if "reference_emg_profile" in source.files else None
    calibration = fit_session_calibration(
        source["rest_emg"], source["gesture_emg"], source["rest_accel"], source["rest_gyro"],
        source["motion_accel"], source["motion_gyro"],
        observed_direction_vectors=directions or None,
        reference_emg_profile=reference,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_train_baseline(args: argparse.Namespace) -> int:
    trials = discover_trials(args.dataset)
    calibrations = load_calibrations(args.dataset)
    train = build_feature_windows(trials_for_split(trials, "train"), calibrations)
    validation = build_feature_windows(trials_for_split(trials, "validation"), calibrations)
    predictor = BaselinePredictor().fit(
        train.emg, train.imu, train.direction, train.gesture,
        validation_emg_features=validation.emg,
        validation_imu_features=validation.imu,
        validation_direction=validation.direction,
        validation_gesture=validation.gesture,
    )
    lda_gesture, lda_direction = fit_lda_sanity_baselines(
        train.emg, train.imu, train.gesture, train.direction,
    )
    lda_d = lda_direction.predict(validation.imu)
    lda_h = lda_gesture.predict(validation.emg)
    predictor.metadata["lda_validation"] = {
        "direction_macro_f1": _macro_f1(validation.direction, lda_d),
        "gesture_macro_f1": _macro_f1(validation.gesture, lda_h),
        "joint_accuracy": float(np.mean(
            (lda_d == validation.direction) & (lda_h == validation.gesture)
        )),
    }
    predictor.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "train_windows": len(train.emg),
        "validation_windows": len(validation.emg),
        "direction_threshold": predictor.direction_threshold,
        "gesture_threshold": predictor.gesture_threshold,
        "lda_validation": predictor.metadata["lda_validation"],
    }, indent=2))
    return 0


def cmd_train_neural(args: argparse.Namespace) -> int:
    trials = discover_trials(args.dataset)
    calibrations = load_calibrations(args.dataset)
    train = build_raw_windows(trials_for_split(trials, "train"), calibrations)
    validation = build_raw_windows(trials_for_split(trials, "validation"), calibrations)
    result = train_dual_branch(
        train, validation, device=args.device, max_epochs=args.max_epochs,
        patience=args.patience, batch_size=args.batch_size,
    )
    save_neural_artifact(result, args.output)
    print(json.dumps({
        "output": str(args.output),
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
    }, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("Session 4 is locked. Re-run once with --unlock-test only after freezing the model.")
    test_record = Path(args.dataset) / "SESSION4_FINAL_RESULT.json"
    if args.split == "test" and test_record.exists():
        raise SystemExit(f"Session 4 was already evaluated; locked result: {test_record}")
    kind = getattr(args, "kind", "baseline")
    predictor = BaselinePredictor.load(args.model) if kind == "baseline" else load_neural_artifact(args.model)
    trials = discover_trials(args.dataset)
    selected_trials = trials_for_split(trials, args.split)
    calibrations = load_calibrations(args.dataset)
    windows = (
        build_feature_windows(selected_trials, calibrations)
        if kind == "baseline" else build_raw_windows(selected_trials, calibrations)
    )
    d_pred: list[int] = []
    h_pred: list[int] = []
    q_d: list[float] = []
    q_h: list[float] = []
    for emg, imu in zip(windows.emg, windows.imu):
        prediction = predictor.predict_features(emg, imu) if kind == "baseline" else predictor.predict(emg, imu)
        d_pred.append(int(prediction.direction)); h_pred.append(int(prediction.gesture))
        q_d.append(prediction.q_direction); q_h.append(prediction.q_gesture)
    latencies: list[float] = []
    for trial in selected_trials:
        if "arm_onset_ms" not in trial.events and "hand_onset_ms" not in trial.events:
            continue
        estimator = HumanStateEstimator(predictor, calibrations[trial.session_id])
        states = estimator.push_batch(trial.timestamp_ms, trial.emg, trial.accel, trial.gyro)
        timestamps = np.asarray([state.timestamp_ms for state in states])
        if "arm_onset_ms" in trial.events:
            after = np.flatnonzero(trial.timestamp_ms >= trial.events["arm_onset_ms"])
            if len(after):
                target = int(trial.direction[after[0]])
                if target != int(Direction.NONE) and target != int(Direction.UNKNOWN):
                    latency = first_stable_latency(
                        trial.events["arm_onset_ms"], timestamps,
                        np.asarray([int(state.direction) for state in states]), target,
                    )
                    if latency is not None: latencies.append(latency)
        if "hand_onset_ms" in trial.events:
            after = np.flatnonzero(trial.timestamp_ms >= trial.events["hand_onset_ms"])
            if len(after):
                target = int(trial.gesture[after[0]])
                if target != int(Gesture.NEUTRAL) and target != int(Gesture.UNKNOWN):
                    latency = first_stable_latency(
                        trial.events["hand_onset_ms"], timestamps,
                        np.asarray([int(state.gesture) for state in states]), target,
                    )
                    if latency is not None: latencies.append(latency)
    result = evaluate_predictions(
        windows.direction, windows.gesture, np.asarray(d_pred), np.asarray(h_pred),
        np.asarray(q_d), np.asarray(q_h),
        latency_ms=np.asarray(latencies) if latencies else None,
    )
    result_payload = asdict(result)
    if args.split == "test":
        model_path = Path(args.model)
        result_payload = {
            **result_payload,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path.resolve()),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "split": "test",
        }
        test_record.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(json.dumps(result_payload, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    calibration = SessionCalibration.from_dict(json.loads(Path(args.calibration).read_text(encoding="utf-8")))
    predictor = (
        BaselinePredictor.load(args.model)
        if args.kind == "baseline" else load_neural_artifact(args.model, device=args.device)
    )
    consistency = ConditionalConsistencyModel.load(args.consistency) if args.consistency else None
    estimator = HumanStateEstimator(predictor, calibration, consistency_model=consistency)
    publisher = OscPublisher(args.output_host, args.output_port, publish_legacy=args.publish_legacy)
    logger = StateCsvLogger(args.log_csv) if args.log_csv else None
    service = LiveClassifierService(estimator, publisher, logger)
    server = RawOscServer(service.accept, args.input_host, args.input_port)
    print(f"raw OSC {args.input_host}:{args.input_port} -> state OSC {args.output_host}:{args.output_port}")
    try:
        server.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
        if logger is not None:
            logger.close()
    return 0


def cmd_fit_consistency(args: argparse.Namespace) -> int:
    records = []
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("onset_lag_ms", "").strip():
                continue
            key = tuple(int(row[name]) for name in ("direction", "gesture", "arm_phase", "hand_phase"))
            records.append((key, float(row["activation"]), float(row["motion"]), float(row["onset_lag_ms"])))
    model = ConditionalConsistencyModel(min_samples=args.min_samples).fit(records)
    model.save(args.output)
    print(json.dumps({"conditions": len(model.stats), "output": args.output}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMG-IMU human-state tools")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("make-smoke-dataset")
    smoke.add_argument("output")
    smoke.add_argument("--repetitions", type=int, default=1)
    smoke.set_defaults(func=cmd_make_smoke)
    validate = commands.add_parser("validate-dataset")
    validate.add_argument("dataset")
    validate.add_argument("--formal", action="store_true", help="require at least 3 trials per state in every session")
    validate.set_defaults(func=cmd_validate)
    calibration = commands.add_parser("fit-calibration")
    calibration.add_argument("input")
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=cmd_fit_calibration)
    baseline = commands.add_parser("train-baseline")
    baseline.add_argument("dataset")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(func=cmd_train_baseline)
    neural = commands.add_parser("train-neural")
    neural.add_argument("dataset")
    neural.add_argument("--output", required=True)
    neural.add_argument("--device", default="cpu")
    neural.add_argument("--max-epochs", type=int, default=100)
    neural.add_argument("--patience", type=int, default=15)
    neural.add_argument("--batch-size", type=int, default=128)
    neural.set_defaults(func=cmd_train_neural)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--unlock-test", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)
    serve = commands.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    serve.add_argument("--calibration", required=True)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--input-host", default="127.0.0.1")
    serve.add_argument("--input-port", type=int, default=9100)
    serve.add_argument("--output-host", default="127.0.0.1")
    serve.add_argument("--output-port", type=int, default=9000)
    serve.add_argument("--publish-legacy", action="store_true")
    serve.add_argument("--consistency", help="optional fitted shadow consistency JSON")
    serve.add_argument("--log-csv", help="append runtime state and onset-lag observations")
    serve.set_defaults(func=cmd_serve)
    consistency = commands.add_parser("fit-consistency")
    consistency.add_argument("input", help="CSV of labeled A/M/onset-lag observations")
    consistency.add_argument("--output", required=True)
    consistency.add_argument("--min-samples", type=int, default=12)
    consistency.set_defaults(func=cmd_fit_consistency)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
