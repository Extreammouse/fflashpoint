"""Webcam gaze tracker for the shipped L2CS runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from src.gaze_model import (
    EmaPointSmoother,
    GazeEstimate,
    L2CSAngleGazeModel,
    L2CSLocalKnnGazeModel,
    load_gaze_model,
)
from src.torch_device import resolve_torch_device


@dataclass
class FaceFeatureSample:
    features: list[float]
    confidence: float
    quality: str
    debug: dict[str, Any]


class L2CSFeatureExtractor:
    """Extract gaze yaw/pitch from one webcam frame with optional L2CS-Net."""

    FEATURE_COUNT = 2

    def __init__(
        self,
        weights_path: str | Path | None = None,
        arch: str | None = None,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path or config.L2CS_WEIGHTS_PATH)
        if not self.weights_path.exists():
            raise RuntimeError(
                "Missing L2CS weights. Expected "
                f"{self.weights_path}. Place L2CSNet_gaze360.pkl under models."
            )

        try:
            import torch
            from l2cs import Pipeline
        except ImportError as exc:  # pragma: no cover - setup dependent.
            raise RuntimeError(
                "Missing L2CS dependencies. Install requirements.txt."
            ) from exc

        self.arch = arch or config.L2CS_ARCH
        self.device = torch.device(resolve_torch_device(device or config.L2CS_DEVICE))
        self._pipeline = Pipeline(
            weights=self.weights_path,
            arch=self.arch,
            device=self.device,
        )

    def close(self) -> None:
        return None

    def extract(self, frame_bgr: np.ndarray) -> FaceFeatureSample | None:
        try:
            results = self._pipeline.step(frame_bgr)
        except ValueError as exc:
            if "need at least one array to stack" in str(exc):
                return None
            raise
        rows = _extract_l2cs_result_rows(results)
        if not rows:
            return None

        row = rows[0]
        yaw = float(row["yaw"])
        pitch = float(row["pitch"])
        confidence = row["confidence"]
        confidence_value = (
            config.L2CS_BASE_CONFIDENCE if confidence is None else float(confidence)
        )
        confidence_value = float(np.clip(confidence_value, 0.2, 1.0))
        debug = {
            "backend": "l2cs",
            "l2cs_result_count": len(rows),
            "l2cs_yaw_rad": round(yaw, 6),
            "l2cs_pitch_rad": round(pitch, 6),
            "l2cs_yaw_deg": round(float(np.degrees(yaw)), 3),
            "l2cs_pitch_deg": round(float(np.degrees(pitch)), 3),
            "l2cs_face_confidence": None
            if confidence is None
            else round(float(confidence), 3),
            "weights": str(self.weights_path),
            "arch": self.arch,
            "device": str(self.device),
        }
        return FaceFeatureSample(
            features=[yaw, pitch],
            confidence=confidence_value,
            quality="l2cs_ok",
            debug=debug,
        )


class WebcamGazeTracker:
    """Live webcam gaze provider using a saved L2CS calibration model."""

    def __init__(
        self,
        camera_index: int,
        calibration_path: str | Path,
        smoothing_alpha: float,
    ) -> None:
        self.camera_index = camera_index
        self.calibration_path = Path(calibration_path)
        self.model = load_gaze_model(self.calibration_path)
        accepted_models = {
            L2CSAngleGazeModel.model_type,
            L2CSLocalKnnGazeModel.model_type,
        }
        if getattr(self.model, "model_type", "") not in accepted_models:
            raise RuntimeError(
                "Calibration model is unsupported by the shipped runtime. "
                "Rerun calibrate_gaze.py with the default dense L2CS profile."
            )
        self.backend = "l2cs"
        self.extractor = L2CSFeatureExtractor(config.L2CS_WEIGHTS_PATH)
        if self.model.feature_count > self.extractor.FEATURE_COUNT:
            self.extractor.close()
            raise RuntimeError(
                "Calibration feature format is outdated. Rerun calibrate_gaze.py "
                f"(model requires {self.model.feature_count}, extractor has {self.extractor.FEATURE_COUNT})."
            )
        self.capture = open_camera(camera_index)
        if not self.capture.isOpened():
            self.extractor.close()
            raise RuntimeError(f"Could not open webcam at camera index {camera_index}.")

        alpha, median_window, max_step_px = self._smoothing_settings(smoothing_alpha)
        self.smoothing_alpha = alpha
        self.smoothing_median_window = median_window
        self.smoothing_max_step_px = max_step_px
        self.smoother = EmaPointSmoother(
            alpha,
            median_window=median_window,
            max_step_px=max_step_px,
        )
        self.last_status = "webcam ready"
        self.last_debug: dict[str, Any] = {}

    @classmethod
    def calibration_exists(cls, calibration_path: str | Path) -> bool:
        return Path(calibration_path).exists()

    def read(self) -> GazeEstimate | None:
        ok, frame = self.capture.read()
        if not ok:
            self.last_status = "webcam frame unavailable"
            self.last_debug = {"status": self.last_status}
            self.smoother.reset()
            return None

        try:
            sample = self.extractor.extract(frame)
        except Exception as exc:
            if config.NO_FACE_CENTER_FALLBACK:
                return self._center_fallback_estimate(
                    status=f"tracker extract error: {type(exc).__name__}",
                    quality="tracker_error_center_fallback",
                    debug={
                        "status": "tracker extract error",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    },
                )
            raise
        if sample is None:
            if config.NO_FACE_CENTER_FALLBACK:
                return self._center_fallback_estimate(
                    status="face/eyes not found; center fallback",
                    quality="no_face_center_fallback",
                    debug={"status": "face/eyes not found"},
                )
            self.last_status = "face/eyes not found"
            self.last_debug = {"status": self.last_status}
            self.smoother.reset()
            return None

        try:
            prediction_debug: dict[str, Any] = {}
            if hasattr(self.model, "predict_with_diagnostics"):
                raw_point, prediction_debug = self.model.predict_with_diagnostics(
                    sample.features
                )
            else:
                raw_point = self.model.predict(sample.features)
        except ValueError as exc:
            self.last_status = f"calibration incompatible: {exc}"
            self.smoother.reset()
            return None

        gaze_x, gaze_y = self.smoother.update(raw_point)
        quality = self._quality_label(sample.quality, prediction_debug)
        confidence = self._calibrated_confidence(sample.confidence, prediction_debug)
        self.last_status = quality
        self.last_debug = {
            "status": self.last_status,
            "raw_point": {"x": round(raw_point[0], 2), "y": round(raw_point[1], 2)},
            "smoothed_point": {"x": round(gaze_x, 2), "y": round(gaze_y, 2)},
            "face_confidence": round(float(sample.confidence), 3),
            "calibrated_confidence": round(float(confidence), 3),
            "validation_error_px": self._effective_validation_error(),
            "raw_validation_error_px": self.model.validation_error_px,
            "smoothing_alpha": self.smoothing_alpha,
            "median_window": self.smoothing_median_window,
            "max_step_px": self.smoothing_max_step_px,
            "prediction": prediction_debug,
            "sample": sample.debug,
        }
        return GazeEstimate(
            x=gaze_x,
            y=gaze_y,
            confidence=confidence,
            source=self._source_name(),
            quality=quality,
        )

    def calibration_summary(self) -> dict[str, Any]:
        return {
            "available": True,
            "average_error_px": self._effective_validation_error(),
            "validation_error_px": self.model.validation_error_px,
            "training_error_px": self.model.training_error_px,
            "sample_count": self.model.sample_count,
            "point_errors_px": self.model.point_errors_px,
            "model_type": getattr(self.model, "model_type", "linear_ridge_gaze_v1"),
            "backend": self._backend_name(),
            "x_signal_range": getattr(self.model, "x_signal_range", None),
            "y_signal_range": getattr(self.model, "y_signal_range", None),
        }

    def close(self) -> None:
        self.capture.release()
        self.extractor.close()

    def _calibrated_confidence(
        self,
        face_confidence: float,
        prediction_debug: dict[str, Any] | None = None,
    ) -> float:
        error = self._effective_validation_error()
        confidence = float(face_confidence)
        if error is None:
            pass
        elif error >= config.CALIBRATION_POOR_ERROR_PX:
            confidence *= 0.35
        elif error >= config.CALIBRATION_GOOD_ERROR_PX:
            confidence *= 0.65
        if prediction_debug and prediction_debug.get("out_of_calibration_range"):
            confidence *= 0.5
        return confidence

    def _quality_label(
        self,
        base_quality: str,
        prediction_debug: dict[str, Any] | None = None,
    ) -> str:
        parts = [base_quality]
        error = self._effective_validation_error()
        if error is not None and error >= config.CALIBRATION_POOR_ERROR_PX:
            parts.append("poor_calibration")
        elif error is not None and error >= config.CALIBRATION_GOOD_ERROR_PX:
            parts.append("fair_calibration")
        x_range = getattr(self.model, "x_signal_range", config.GAZE_MIN_SIGNAL_RANGE)
        y_range = getattr(self.model, "y_signal_range", config.GAZE_MIN_SIGNAL_RANGE)
        if x_range < config.GAZE_MIN_SIGNAL_RANGE or y_range < config.GAZE_MIN_SIGNAL_RANGE:
            parts.append("weak_gaze_signal")
        elif error is not None and error < config.CALIBRATION_GOOD_ERROR_PX:
            parts.append("good_calibration")
        if prediction_debug and prediction_debug.get("out_of_calibration_range"):
            parts.append("outside_calibration_range")
        return ";".join(parts)

    def _backend_name(self) -> str:
        return "l2cs"

    def _source_name(self) -> str:
        return "webcam_calibrated"

    def _smoothing_settings(self, fallback_alpha: float) -> tuple[float, int, float | None]:
        return (
            config.L2CS_GAZE_SMOOTHING_ALPHA,
            config.L2CS_GAZE_MEDIAN_WINDOW,
            config.L2CS_GAZE_MAX_STEP_PX,
        )

    def _effective_validation_error(self) -> float | None:
        error = self.model.validation_error_px
        if _is_finite_number(error):
            return float(error)
        training_error = self.model.training_error_px
        if _is_finite_number(training_error):
            return float(training_error)
        return None

    def _center_fallback_estimate(
        self,
        *,
        status: str,
        quality: str,
        debug: dict[str, Any] | None = None,
    ) -> GazeEstimate:
        self.smoother.reset()
        screen_width, screen_height = self.model.screen_size
        center_x = float(screen_width) / 2.0
        center_y = float(screen_height) / 2.0
        confidence = float(config.NO_FACE_FALLBACK_CONFIDENCE)
        self.last_status = status
        self.last_debug = {
            "status": self.last_status,
            "fallback": "screen_center",
            "raw_point": {"x": round(center_x, 2), "y": round(center_y, 2)},
            "smoothed_point": {"x": round(center_x, 2), "y": round(center_y, 2)},
            "face_confidence": 0.0,
            "calibrated_confidence": round(confidence, 3),
            "validation_error_px": self._effective_validation_error(),
            "raw_validation_error_px": self.model.validation_error_px,
            "smoothing_alpha": self.smoothing_alpha,
            "median_window": self.smoothing_median_window,
            "max_step_px": self.smoothing_max_step_px,
            "prediction": {"fallback": True},
            "sample": debug or {},
        }
        return GazeEstimate(
            x=center_x,
            y=center_y,
            confidence=confidence,
            source="webcam_center_fallback",
            quality=quality,
        )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a webcam with Windows-friendly backend fallbacks."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture()


def _extract_l2cs_result_rows(results: Any) -> list[dict[str, float | None]]:
    rows = []
    for item in _iter_l2cs_results(results):
        yaw_values = _read_l2cs_values(item, "yaw")
        pitch_values = _read_l2cs_values(item, "pitch")
        pair_count = min(len(yaw_values), len(pitch_values))
        if pair_count <= 0:
            continue
        confidence_values = _read_l2cs_nested_values(item, ("face", "confidence"))
        if not confidence_values:
            confidence_values = [None] * pair_count
        elif len(confidence_values) == 1 and pair_count > 1:
            confidence_values = confidence_values * pair_count
        elif len(confidence_values) < pair_count:
            confidence_values.extend([confidence_values[-1]] * (pair_count - len(confidence_values)))

        for index in range(pair_count):
            rows.append(
                {
                    "yaw": yaw_values[index],
                    "pitch": pitch_values[index],
                    "confidence": confidence_values[index] if index < len(confidence_values) else None,
                }
            )
    return rows


def _iter_l2cs_results(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, dict):
        return [results]
    try:
        return list(results)
    except TypeError:
        return [results]


def _read_l2cs_values(item: Any, name: str) -> list[float]:
    if isinstance(item, dict):
        return _to_float_list(item.get(name))
    return _to_float_list(getattr(item, name, None))


def _read_l2cs_nested_values(item: Any, path: tuple[str, ...]) -> list[float | None]:
    current = item
    for index, key in enumerate(path):
        if isinstance(current, (list, tuple)):
            values: list[float | None] = []
            remaining_path = path[index:]
            for entry in current:
                values.extend(_read_l2cs_nested_values(entry, remaining_path))
            return values
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return []
    return [value for value in _to_optional_float_list(current)]


def _to_float_list(value: Any) -> list[float]:
    return [item for item in _to_optional_float_list(value) if item is not None]


def _to_optional_float_list(value: Any) -> list[float | None]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return _to_optional_float_list(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        values: list[float | None] = []
        for item in value:
            values.extend(_to_optional_float_list(item))
        return values
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return [None]


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
