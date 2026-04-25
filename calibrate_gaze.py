"""L2CS-based webcam gaze calibration.

This script stores only calibration parameters in artifacts/calibration.json.
It does not save webcam frames, screenshots, video, or image files.
"""

from __future__ import annotations

import ctypes
import argparse
import json
from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from src.gaze_model import (
    L2CSAngleGazeModel,
    L2CSLocalKnnGazeModel,
)
from src.gaze_tracker import L2CSFeatureExtractor, open_camera


WINDOW_NAME = "Gaze Calibration"


@dataclass(frozen=True)
class CalibrationTarget:
    point: tuple[float, float]
    emphasis: str
    stage: str
    sample_scale: float = 1.0


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    targets: list[CalibrationTarget]
    selection_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the shipped L2CS webcam gaze models.")
    parser.add_argument(
        "--profile",
        choices=("standard3x3", "dense5x5"),
        default=config.GAZE_CALIBRATION_PROFILE,
        help=(
            "Calibration target layout. standard3x3 keeps the faster legacy grid. "
            "dense5x5 adds a denser 5x5 grid plus a short corner-refinement pass."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("l2cs",),
        default="l2cs",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        choices=(
            "auto",
            "l2cs_angle_linear",
            "l2cs_angle_poly2",
            "l2cs_local_knn",
        ),
        default=config.GAZE_CALIBRATION_MODEL,
        help="Calibration model to save. auto selects the best validation score for the chosen profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen_size = get_primary_monitor_size()
    profile = calibration_profile(screen_size, args.profile)
    points = [target.point for target in profile.targets]
    capture = open_camera(config.CAMERA_INDEX)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam at camera index {config.CAMERA_INDEX}.")

    extractor = create_feature_extractor()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        warm_up_extractor(
            capture=capture,
            extractor=extractor,
            screen_size=screen_size,
            seconds=config.L2CS_CALIBRATION_WARMUP_SECONDS,
        )
        training_features: list[list[float]] = []
        training_targets: list[tuple[float, float]] = []
        raw_training_features: list[list[float]] = []
        raw_training_targets: list[tuple[float, float]] = []
        point_summaries: list[dict[str, Any]] = []
        total_raw_samples = 0

        for index, target_spec in enumerate(profile.targets, start=1):
            samples = collect_samples_for_point(
                capture=capture,
                extractor=extractor,
                screen_size=screen_size,
                target=target_spec.point,
                label=calibration_point_label(index, len(profile.targets), target_spec),
                sample_seconds=sample_seconds_for_target(target_spec),
                max_sample_seconds=max_sample_seconds_for_calibration(),
                settle_seconds=config.CALIBRATION_SETTLE_SECONDS,
                min_samples=min_samples_for_calibration(),
            )
            if samples is None:
                raise SystemExit("Calibration cancelled.")
            summary = summarize_samples(samples, min_samples_for_calibration())
            if summary is None:
                raise SystemExit(
                    f"Not enough stable face/eye samples for point {index}. "
                    "Keep your face visible and rerun calibration."
                )
            training_features.append(summary["median_features"])
            training_targets.append(target_spec.point)
            raw_training_features.extend(samples)
            raw_training_targets.extend([target_spec.point] * len(samples))
            total_raw_samples += int(summary["raw_count"])
            point_summaries.append(
                build_point_summary(
                    target_spec.point,
                    summary["median_features"],
                    summary,
                    target_spec,
                )
            )

        if len(training_features) < 6:
            raise SystemExit("Not enough face/eye samples collected for calibration.")

        validation_features = collect_validation_features(
            capture=capture,
            extractor=extractor,
            screen_size=screen_size,
            targets=profile.targets,
        )
        if validation_features is None:
            raise SystemExit("Validation cancelled.")
        candidates = build_candidate_models(
            training_features,
            training_targets,
            raw_training_features,
            raw_training_targets,
            screen_size=screen_size,
            profile_name=profile.name,
            requested_model=args.model,
        )
        selected_name, model, validation_average, point_errors, candidate_results = (
            select_calibration_model(
                candidates=candidates,
                requested_model=args.model,
                validation_features=validation_features,
                validation_targets=points,
                profile=profile,
            )
        )
        model.sample_count = total_raw_samples
        model.point_errors_px = point_errors
        model.validation_error_px = validation_average
        model.save(config.CALIBRATION_PATH)
        write_calibration_diagnostics(
            screen_size=screen_size,
            point_summaries=point_summaries,
            feature_count=len(training_features[0]),
            training_error=model.training_error_px,
            validation_average=validation_average,
            point_errors=point_errors,
            selected_model=selected_name,
            candidate_results=candidate_results,
            signal_stats=get_signal_stats(model, training_features),
            calibration_path=Path(config.CALIBRATION_PATH),
            profile_name=profile.name,
            selection_mode=profile.selection_mode,
        )
        show_done(
            screen_size,
            model.training_error_px,
            validation_average,
            Path(config.CALIBRATION_PATH),
            point_errors,
        )
        print(f"Calibration saved to {config.CALIBRATION_PATH}")
        print(f"Selected model: {selected_name} ({getattr(model, 'model_type', type(model).__name__)})")
        print(f"Training error: {model.training_error_px:.1f}px")
        print(f"Validation average error: {format_error(validation_average)}")
        print(f"Validation point errors: {[format_error(value) for value in point_errors]}")
    finally:
        capture.release()
        extractor.close()
        cv2.destroyAllWindows()


def collect_samples_for_point(
    capture: cv2.VideoCapture,
    extractor: Any,
    screen_size: tuple[int, int],
    target: tuple[float, float],
    label: str,
    sample_seconds: float,
    max_sample_seconds: float,
    settle_seconds: float,
    min_samples: int,
) -> list[list[float]] | None:
    started_at = time.monotonic()
    samples: list[list[float]] = []
    total_seconds = settle_seconds + sample_seconds
    max_total_seconds = settle_seconds + max_sample_seconds

    while True:
        elapsed = time.monotonic() - started_at
        if _should_finish_collection(
            elapsed=elapsed,
            total_seconds=total_seconds,
            max_total_seconds=max_total_seconds,
            sample_count=len(samples),
            min_samples=min_samples,
        ):
            return samples

        ok, frame = capture.read()
        sample = extractor.extract(frame) if ok else None
        collecting = elapsed >= settle_seconds
        if collecting and sample is not None:
            samples.append(sample.features)

        canvas = calibration_canvas(
            screen_size=screen_size,
            target=target,
            title=label,
            subtitle=calibration_subtitle(),
            sample_count=len(samples),
            min_samples=min_samples,
            collecting=collecting,
            seconds_left=max(0.0, total_seconds - elapsed),
            extra_seconds_left=max(0.0, max_total_seconds - elapsed),
            face_ok=sample is not None,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None


def collect_validation_features(
    capture: cv2.VideoCapture,
    extractor: Any,
    screen_size: tuple[int, int],
    targets: list[CalibrationTarget],
) -> list[list[float]] | None:
    validation_features: list[list[float]] = []
    total_targets = len(targets)
    for index, target_spec in enumerate(targets, start=1):
        samples = collect_samples_for_point(
            capture=capture,
            extractor=extractor,
            screen_size=screen_size,
            target=target_spec.point,
            label=validation_point_label(index, total_targets, target_spec),
            sample_seconds=validation_seconds_for_target(target_spec),
            max_sample_seconds=max_sample_seconds_for_calibration(),
            settle_seconds=config.CALIBRATION_SETTLE_SECONDS,
            min_samples=min_samples_for_calibration(),
        )
        if samples is None:
            return None
        summary = summarize_samples(samples, min_samples_for_calibration())
        if summary is None:
            validation_features.append([])
            continue
        validation_features.append(summary["median_features"])

    return validation_features


def build_candidate_models(
    training_features: list[list[float]],
    training_targets: list[tuple[float, float]],
    raw_training_features: list[list[float]],
    raw_training_targets: list[tuple[float, float]],
    screen_size: tuple[int, int],
    profile_name: str,
    requested_model: str,
) -> dict[str, Any]:
    candidates = {
        "l2cs_angle_linear": L2CSAngleGazeModel.fit(
            raw_training_features,
            raw_training_targets,
            screen_size=screen_size,
            degree=1,
            ridge_lambda=config.L2CS_RIDGE_LAMBDA,
            range_margin=config.L2CS_OUT_OF_RANGE_MARGIN,
        ),
        "l2cs_angle_poly2": L2CSAngleGazeModel.fit(
            raw_training_features,
            raw_training_targets,
            screen_size=screen_size,
            degree=2,
            ridge_lambda=config.L2CS_RIDGE_LAMBDA,
            range_margin=config.L2CS_OUT_OF_RANGE_MARGIN,
        ),
    }
    if profile_name == "dense5x5" or requested_model == "l2cs_local_knn":
        candidates["l2cs_local_knn"] = L2CSLocalKnnGazeModel.fit(
            raw_training_features,
            raw_training_targets,
            screen_size=screen_size,
            neighbor_count=config.L2CS_LOCAL_KNN_NEIGHBORS,
            range_margin=config.L2CS_LOCAL_KNN_OUT_OF_RANGE_MARGIN,
        )
    return candidates


def select_calibration_model(
    candidates: dict[str, Any],
    requested_model: str,
    validation_features: list[list[float]],
    validation_targets: list[tuple[float, float]],
    profile: CalibrationProfile,
) -> tuple[str, Any, float | None, list[float | None], dict[str, Any]]:
    candidate_results: dict[str, Any] = {}
    for name, model in candidates.items():
        average, errors = evaluate_model(model, validation_features, validation_targets)
        candidate_results[name] = {
            "model_type": getattr(model, "model_type", type(model).__name__),
            "training_error_px": model.training_error_px,
            "validation_average_error_px": average,
            "point_errors_px": errors,
        }

    if requested_model == "auto":
        selected_name = min(
            candidate_results,
            key=lambda name: _candidate_selection_score(
                candidate_results[name],
                profile.selection_mode,
                profile.targets,
            ),
        )
    else:
        selected_name = requested_model
        if selected_name not in candidates:
            raise ValueError(f"Unknown calibration model: {requested_model}")

    selected = candidates[selected_name]
    selected_result = candidate_results[selected_name]
    return (
        selected_name,
        selected,
        selected_result["validation_average_error_px"],
        selected_result["point_errors_px"],
        candidate_results,
    )


def evaluate_model(
    model: Any,
    validation_features: list[list[float]],
    validation_targets: list[tuple[float, float]],
) -> tuple[float | None, list[float | None]]:
    errors: list[float | None] = []
    for features, target in zip(validation_features, validation_targets):
        if not features:
            errors.append(None)
            continue
        predicted = model.predict(features)
        errors.append(math.dist(predicted, target))
    finite_errors = [value for value in errors if is_finite_number(value)]
    if not finite_errors:
        return None, errors
    return float(np.mean(finite_errors)), errors


def _candidate_selection_score(
    result: dict[str, Any],
    selection_mode: str,
    targets: list[CalibrationTarget],
) -> tuple[Any, ...]:
    validation_error = result["validation_average_error_px"]
    if is_finite_number(validation_error):
        point_errors = result.get("point_errors_px", [])
        finite_errors = [float(value) for value in point_errors if is_finite_number(value)]
        if selection_mode == "corner_aware":
            corner_errors = [
                float(point_errors[index])
                for index, target in enumerate(targets)
                if target.emphasis == "corner"
                and index < len(point_errors)
                and is_finite_number(point_errors[index])
            ]
            if corner_errors:
                return (
                    0,
                    max(corner_errors),
                    float(np.mean(corner_errors)),
                    float(validation_error),
                )
        if finite_errors:
            return (0, float(validation_error), max(finite_errors))
        return (0, float(validation_error))
    training_error = result.get("training_error_px")
    if is_finite_number(training_error):
        return (1, float(training_error))
    return (2, float("inf"))


def get_signal_stats(model: Any, feature_rows: list[list[float]]) -> dict[str, float]:
    signal_stats = getattr(model, "signal_stats", None)
    if isinstance(signal_stats, dict) and signal_stats:
        return {key: float(value) for key, value in signal_stats.items()}

    features = np.asarray(feature_rows, dtype=np.float64)
    stats = {
        "x_signal_range": float(np.max(features[:, 0]) - np.min(features[:, 0])),
        "y_signal_range": float(np.max(features[:, 1]) - np.min(features[:, 1])),
    }
    if features.shape[1] >= 16:
        stats["yaw_range_deg"] = float(np.max(features[:, 14]) - np.min(features[:, 14]))
        stats["pitch_range_deg"] = float(
            np.max(features[:, 15]) - np.min(features[:, 15])
        )
    return stats


def summarize_samples(
    samples: list[list[float]],
    min_samples: int | None = None,
) -> dict[str, Any] | None:
    """Return a robust median feature row for one calibration target."""
    min_count = config.CALIBRATION_MIN_SAMPLES_PER_POINT if min_samples is None else min_samples
    if len(samples) < min_count:
        return None

    values = np.asarray(samples, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust_scale = np.where(mad < 1e-6, 1.0, 1.4826 * mad)
    robust_z = np.abs((values - median) / robust_scale)
    row_scores = np.median(robust_z, axis=1)
    keep_mask = row_scores <= 3.5

    if int(np.count_nonzero(keep_mask)) < min_count:
        keep_count = max(min_count, len(values) // 2)
        keep_indices = np.argsort(row_scores)[:keep_count]
        kept = values[keep_indices]
    else:
        kept = values[keep_mask]

    return {
        "median_features": np.median(kept, axis=0).astype(float).tolist(),
        "raw_count": len(samples),
        "kept_count": int(len(kept)),
    }


def write_calibration_diagnostics(
    screen_size: tuple[int, int],
    point_summaries: list[dict[str, Any]],
    feature_count: int,
    training_error: float | None,
    validation_average: float | None,
    point_errors: list[float | None],
    selected_model: str,
    candidate_results: dict[str, Any],
    signal_stats: dict[str, float],
    calibration_path: Path,
    profile_name: str,
    selection_mode: str,
) -> None:
    diagnostics_path = Path(config.CALIBRATION_DIAGNOSTICS_PATH)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    points_with_errors = []
    for summary, error in zip(point_summaries, point_errors):
        merged = dict(summary)
        merged["validation_error_px"] = round_optional(error)
        points_with_errors.append(merged)
    finite_point_errors = [value for value in point_errors if is_finite_number(value)]

    payload = {
        "timestamp": round(time.time(), 6),
        "screen": {"width": screen_size[0], "height": screen_size[1]},
        "calibration_path": str(calibration_path),
        "model": {
            "backend": "l2cs",
            "profile": profile_name,
            "selection_mode": selection_mode,
            "selected": selected_model,
            "type": candidate_results[selected_model]["model_type"],
            "feature_count": feature_count,
            "ridge_lambda": config.L2CS_RIDGE_LAMBDA,
            "knn_neighbors": config.L2CS_LOCAL_KNN_NEIGHBORS,
            "knn_out_of_range_margin": config.L2CS_LOCAL_KNN_OUT_OF_RANGE_MARGIN,
            "signal_stats": {
                key: round(float(value), 6) for key, value in signal_stats.items()
            },
            "min_signal_range": config.GAZE_MIN_SIGNAL_RANGE,
        },
        "candidate_results": _json_safe_candidate_results(candidate_results),
        "sampling": {
            "sample_seconds": sample_seconds_for_calibration(),
            "settle_seconds": config.CALIBRATION_SETTLE_SECONDS,
            "validation_seconds": validation_seconds_for_calibration(),
            "min_samples_per_point": min_samples_for_calibration(),
        },
        "quality": {
            "training_error_px": round_optional(training_error),
            "validation_average_error_px": round_optional(validation_average),
            "validation_worst_point_error_px": round_optional(max(finite_point_errors))
            if finite_point_errors
            else None,
            "validation_points_available": len(finite_point_errors),
            "validation_points_expected": len(point_errors),
            "good_error_px": config.CALIBRATION_GOOD_ERROR_PX,
            "poor_error_px": config.CALIBRATION_POOR_ERROR_PX,
        },
        "points": points_with_errors,
    }
    diagnostics_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _json_safe_candidate_results(candidate_results: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for name, result in candidate_results.items():
        safe[name] = {
            "model_type": result["model_type"],
            "training_error_px": None
            if result["training_error_px"] is None
            else round_optional(result["training_error_px"]),
            "validation_average_error_px": round_optional(
                result["validation_average_error_px"]
            ),
            "point_errors_px": [round_optional(value) for value in result["point_errors_px"]],
        }
    return safe


def sample_seconds_for_calibration() -> float:
    return config.L2CS_CALIBRATION_SAMPLE_SECONDS


def sample_seconds_for_target(target_spec: CalibrationTarget) -> float:
    return sample_seconds_for_calibration() * float(target_spec.sample_scale)


def validation_seconds_for_calibration() -> float:
    return config.L2CS_CALIBRATION_VALIDATION_SECONDS


def validation_seconds_for_target(target_spec: CalibrationTarget) -> float:
    return validation_seconds_for_calibration() * float(target_spec.sample_scale)


def min_samples_for_calibration() -> int:
    return config.L2CS_CALIBRATION_MIN_SAMPLES_PER_POINT


def max_sample_seconds_for_calibration() -> float:
    return config.L2CS_CALIBRATION_MAX_SAMPLE_SECONDS


def warm_up_extractor(
    capture: cv2.VideoCapture,
    extractor: Any,
    screen_size: tuple[int, int],
    seconds: float,
) -> None:
    if seconds <= 0:
        return

    started_at = time.monotonic()
    while True:
        elapsed = time.monotonic() - started_at
        if elapsed >= seconds:
            return

        ok, frame = capture.read()
        if ok:
            try:
                extractor.extract(frame)
            except Exception:
                pass

        canvas = np.zeros((screen_size[1], screen_size[0], 3), dtype=np.uint8)
        canvas[:, :] = (18, 20, 24)
        draw_lines(
            canvas,
            [
                "Preparing gaze model",
                "Keep your face visible. Warming up webcam gaze inference.",
                f"{max(0.0, seconds - elapsed):.1f}s left",
                "No webcam frames or screen images are saved.",
            ],
        )
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)


def _should_finish_collection(
    *,
    elapsed: float,
    total_seconds: float,
    max_total_seconds: float,
    sample_count: int,
    min_samples: int,
) -> bool:
    if elapsed < total_seconds:
        return False
    if sample_count >= min_samples:
        return True
    return elapsed >= max_total_seconds


def is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def round_optional(value: Any, digits: int = 2) -> float | None:
    if not is_finite_number(value):
        return None
    return round(float(value), digits)


def format_error(value: Any) -> str:
    rounded = round_optional(value, 1)
    if rounded is None:
        return "n/a"
    return f"{rounded:.1f}px"


def calibration_profile(
    screen_size: tuple[int, int],
    profile_name: str,
) -> CalibrationProfile:
    width, height = screen_size
    if profile_name == "dense5x5":
        grid_fractions = tuple(float(value) for value in config.CALIBRATION_DENSE_GRID_FRACTIONS)
        targets = _grid_targets(width, height, grid_fractions, stage="grid")
        refinement_targets = [
            CalibrationTarget(
                point=(x_fraction * width, y_fraction * height),
                emphasis="corner",
                stage="refinement",
                sample_scale=config.CALIBRATION_REFINEMENT_SAMPLE_SCALE,
            )
            for x_fraction, y_fraction in config.CALIBRATION_DENSE_REFINEMENT_CORNERS
        ]
        return CalibrationProfile(
            name="dense5x5",
            targets=targets + refinement_targets,
            selection_mode="corner_aware",
        )

    grid_fractions = tuple(float(value) for value in config.CALIBRATION_STANDARD_GRID_FRACTIONS)
    return CalibrationProfile(
        name="standard3x3",
        targets=_grid_targets(width, height, grid_fractions, stage="grid"),
        selection_mode="mean",
    )


def calibration_point_label(
    index: int,
    total: int,
    target_spec: CalibrationTarget,
) -> str:
    if target_spec.stage == "refinement":
        return f"Corner refinement {index}/{total}"
    return f"Calibration {index}/{total}"


def validation_point_label(
    index: int,
    total: int,
    target_spec: CalibrationTarget,
) -> str:
    if target_spec.stage == "refinement":
        return f"Validation corner {index}/{total}"
    return f"Validation {index}/{total}"


def _grid_targets(
    width: int,
    height: int,
    fractions: tuple[float, ...],
    *,
    stage: str,
) -> list[CalibrationTarget]:
    targets: list[CalibrationTarget] = []
    last_index = len(fractions) - 1
    for y_index, y_fraction in enumerate(fractions):
        for x_index, x_fraction in enumerate(fractions):
            if x_index in (0, last_index) and y_index in (0, last_index):
                emphasis = "corner"
            elif x_index in (0, last_index) or y_index in (0, last_index):
                emphasis = "edge"
            elif x_index == len(fractions) // 2 and y_index == len(fractions) // 2:
                emphasis = "center"
            else:
                emphasis = "interior"
            targets.append(
                CalibrationTarget(
                    point=(x_fraction * width, y_fraction * height),
                    emphasis=emphasis,
                    stage=stage,
                )
            )
    return targets


def create_feature_extractor() -> Any:
    return L2CSFeatureExtractor(config.L2CS_WEIGHTS_PATH)


def calibration_subtitle() -> str:
    return (
        "Look at the dot with your eyes. Keep your head and shoulders steady. "
        "Press Esc or q to cancel."
    )


def build_point_summary(
    point: tuple[float, float],
    features: list[float],
    summary: dict[str, Any],
    target_spec: CalibrationTarget | None = None,
) -> dict[str, Any]:
    payload = {
        "target": {"x": round(point[0], 2), "y": round(point[1], 2)},
        "raw_count": summary["raw_count"],
        "kept_count": summary["kept_count"],
        "signal_x": round(float(features[0]), 6),
        "signal_y": round(float(features[1]), 6),
    }
    if target_spec is not None:
        payload["stage"] = target_spec.stage
        payload["emphasis"] = target_spec.emphasis
    payload.update(
        {
            "l2cs_yaw_rad": round(float(features[0]), 6),
            "l2cs_pitch_rad": round(float(features[1]), 6),
            "l2cs_yaw_deg": round(float(np.degrees(features[0])), 3),
            "l2cs_pitch_deg": round(float(np.degrees(features[1])), 3),
        }
    )
    return payload


def calibration_canvas(
    screen_size: tuple[int, int],
    target: tuple[float, float],
    title: str,
    subtitle: str,
    sample_count: int,
    min_samples: int,
    collecting: bool,
    seconds_left: float,
    extra_seconds_left: float,
    face_ok: bool,
) -> np.ndarray:
    width, height = screen_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (18, 20, 24)

    target_color = (0, 220, 255) if collecting else (0, 150, 255)
    center = (int(round(target[0])), int(round(target[1])))
    cv2.circle(canvas, center, 34, target_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 10, target_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, 3, (255, 255, 255), -1, cv2.LINE_AA)

    state = "collecting" if collecting else "settling"
    face = "face ok" if face_ok else "face not found"
    lines = [
        title,
        subtitle,
        (
            f"{state} | preferred={seconds_left:.1f}s | max={extra_seconds_left:.1f}s | "
            f"samples={sample_count}/{min_samples} | {face}"
        ),
        "No webcam frames or screen images are saved.",
    ]
    draw_lines(canvas, lines)
    return canvas


def show_done(
    screen_size: tuple[int, int],
    training_error: float | None,
    validation_error: float | None,
    calibration_path: Path,
    point_errors: list[float | None],
) -> None:
    canvas = np.zeros((screen_size[1], screen_size[0], 3), dtype=np.uint8)
    canvas[:, :] = (18, 24, 20)
    finite_point_errors = [value for value in point_errors if is_finite_number(value)]
    lines = [
        "Calibration saved",
        f"Path: {calibration_path}",
        f"Training error: {format_error(training_error)}",
        f"Validation average error: {format_error(validation_error)}",
        "Worst point error: "
        + (format_error(max(finite_point_errors)) if finite_point_errors else "n/a"),
        "Press any key to close.",
    ]
    draw_lines(canvas, lines)
    cv2.imshow(WINDOW_NAME, canvas)
    cv2.waitKey(0)


def draw_lines(canvas: np.ndarray, lines: list[str]) -> None:
    y = 54
    for index, line in enumerate(lines):
        scale = 0.9 if index == 0 else 0.62
        thickness = 2 if index == 0 else 1
        cv2.putText(
            canvas,
            line,
            (48, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (235, 238, 242),
            thickness,
            cv2.LINE_AA,
        )
        y += 42 if index == 0 else 30


def get_primary_monitor_size() -> tuple[int, int]:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    try:
        width = int(ctypes.windll.user32.GetSystemMetrics(0))
        height = int(ctypes.windll.user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return config.SCREEN_WIDTH_FALLBACK, config.SCREEN_HEIGHT_FALLBACK


if __name__ == "__main__":
    main()
