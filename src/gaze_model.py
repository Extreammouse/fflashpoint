"""Calibration models for mapping L2CS gaze angles to screen gaze points."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GazeEstimate:
    """One screen-space gaze estimate."""

    x: float
    y: float
    confidence: float
    source: str
    quality: str

    def as_event_point(self) -> dict[str, Any]:
        return {
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "quality": self.quality,
        }


def load_gaze_model(path: str | Path) -> "L2CSAngleGazeModel | L2CSLocalKnnGazeModel":
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type", "")
    if model_type == L2CSAngleGazeModel.model_type:
        return L2CSAngleGazeModel.from_payload(payload)
    if model_type == L2CSLocalKnnGazeModel.model_type:
        return L2CSLocalKnnGazeModel.from_payload(payload)
    raise RuntimeError(
        "Unsupported calibration model type. Rerun calibrate_gaze.py with the shipped L2CS path."
    )


def _optional_float_list(values: list[Any]) -> list[float | None]:
    return [None if value is None else float(value) for value in values]


class L2CSAngleGazeModel:
    """Calibrated screen mapping from L2CS yaw/pitch angles.

    L2CS gives a gaze direction, not a screen coordinate. This model keeps the
    screen mapping local to this project by fitting yaw/pitch to the calibration
    dots and persisting only numeric parameters.
    """

    model_type = "l2cs_angle_ridge_gaze_v1"
    feature_count = 2

    def __init__(
        self,
        coefficients: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        screen_size: tuple[int, int],
        degree: int,
        ridge_lambda: float,
        range_margin: float,
        signal_stats: dict[str, float],
        validation_error_px: float | None = None,
        training_error_px: float | None = None,
        sample_count: int = 0,
        point_errors_px: list[float] | None = None,
        created_at: float | None = None,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.feature_min = np.asarray(feature_min, dtype=np.float64)
        self.feature_max = np.asarray(feature_max, dtype=np.float64)
        self.screen_size = screen_size
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.range_margin = float(range_margin)
        self.signal_stats = signal_stats
        self.x_signal_range = float(signal_stats.get("yaw_range_rad", 0.0))
        self.y_signal_range = float(signal_stats.get("pitch_range_rad", 0.0))
        self.validation_error_px = validation_error_px
        self.training_error_px = training_error_px
        self.sample_count = sample_count
        self.point_errors_px = point_errors_px or []
        self.created_at = created_at or time.time()

    @classmethod
    def fit(
        cls,
        feature_rows: list[list[float]],
        targets: list[tuple[float, float]],
        screen_size: tuple[int, int],
        degree: int,
        ridge_lambda: float,
        range_margin: float,
        validation_error_px: float | None = None,
    ) -> "L2CSAngleGazeModel":
        if len(feature_rows) < 6:
            raise ValueError("At least 6 L2CS angle samples are required.")
        if len(feature_rows) != len(targets):
            raise ValueError("Feature and target sample counts must match.")

        features = np.asarray(feature_rows, dtype=np.float64)[:, : cls.feature_count]
        target_values = np.asarray(targets, dtype=np.float64)
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
        normalized = (features - feature_mean) / feature_scale
        basis = cls._basis(normalized, degree)

        identity = np.eye(basis.shape[1], dtype=np.float64)
        identity[-1, -1] = 0.0
        lhs = basis.T @ basis + float(ridge_lambda) * identity
        rhs = basis.T @ target_values
        coefficients = np.linalg.solve(lhs, rhs)
        signal_stats = {
            "yaw_range_rad": float(np.max(features[:, 0]) - np.min(features[:, 0])),
            "pitch_range_rad": float(np.max(features[:, 1]) - np.min(features[:, 1])),
            "yaw_range_deg": float(
                np.degrees(np.max(features[:, 0]) - np.min(features[:, 0]))
            ),
            "pitch_range_deg": float(
                np.degrees(np.max(features[:, 1]) - np.min(features[:, 1]))
            ),
            "yaw_min_rad": float(np.min(features[:, 0])),
            "yaw_max_rad": float(np.max(features[:, 0])),
            "pitch_min_rad": float(np.min(features[:, 1])),
            "pitch_max_rad": float(np.max(features[:, 1])),
        }
        model = cls(
            coefficients=coefficients,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            feature_min=np.min(features, axis=0),
            feature_max=np.max(features, axis=0),
            screen_size=screen_size,
            degree=degree,
            ridge_lambda=ridge_lambda,
            range_margin=range_margin,
            signal_stats=signal_stats,
            validation_error_px=validation_error_px,
            sample_count=len(feature_rows),
        )
        predictions = np.asarray([model.predict(row) for row in features])
        errors = np.linalg.norm(predictions - target_values, axis=1)
        model.training_error_px = float(np.mean(errors))
        return model

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "L2CSAngleGazeModel":
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            feature_min=np.asarray(payload["feature_min"], dtype=np.float64),
            feature_max=np.asarray(payload["feature_max"], dtype=np.float64),
            screen_size=(int(payload["screen_width"]), int(payload["screen_height"])),
            degree=int(payload.get("degree", 1)),
            ridge_lambda=float(payload.get("ridge_lambda", 0.0)),
            range_margin=float(payload.get("range_margin", 0.12)),
            signal_stats={key: float(value) for key, value in payload.get("signal_stats", {}).items()},
            validation_error_px=payload.get("validation_error_px"),
            training_error_px=payload.get("training_error_px"),
            sample_count=int(payload.get("sample_count", 0)),
            point_errors_px=_optional_float_list(payload.get("point_errors_px", [])),
            created_at=payload.get("created_at"),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "created_at": self.created_at,
            "feature_count": self.feature_count,
            "screen_width": self.screen_size[0],
            "screen_height": self.screen_size[1],
            "sample_count": self.sample_count,
            "training_error_px": self.training_error_px,
            "validation_error_px": self.validation_error_px,
            "point_errors_px": self.point_errors_px,
            "degree": self.degree,
            "ridge_lambda": self.ridge_lambda,
            "range_margin": self.range_margin,
            "signal_stats": self.signal_stats,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "coefficients": self.coefficients.tolist(),
        }
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def predict(self, features: list[float] | np.ndarray) -> tuple[float, float]:
        point, _debug = self.predict_with_diagnostics(features)
        return point

    def predict_with_diagnostics(
        self,
        features: list[float] | np.ndarray,
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.shape[0] < self.feature_count:
            raise ValueError(f"Expected at least 2 L2CS angle features, got {feature_array.shape[0]}.")

        angles = feature_array[: self.feature_count]
        normalized = (angles - self.feature_mean) / self.feature_scale
        basis = self._basis(normalized, self.degree)
        raw = basis @ self.coefficients
        width, height = self.screen_size
        clipped = (
            float(np.clip(raw[0], 0, max(0, width - 1))),
            float(np.clip(raw[1], 0, max(0, height - 1))),
        )
        debug = self._diagnostics(angles, raw, clipped)
        return clipped, debug

    def _diagnostics(
        self,
        angles: np.ndarray,
        raw: np.ndarray,
        clipped: tuple[float, float],
    ) -> dict[str, Any]:
        feature_range = np.maximum(self.feature_max - self.feature_min, 1e-6)
        lower = self.feature_min - feature_range * self.range_margin
        upper = self.feature_max + feature_range * self.range_margin
        below = np.maximum(lower - angles, 0.0) / feature_range
        above = np.maximum(angles - upper, 0.0) / feature_range
        excess = below + above
        out_indices = [index for index, value in enumerate(excess) if float(value) > 0.0]
        return {
            "model_kind": self.model_type,
            "degree": self.degree,
            "l2cs_yaw_rad": round(float(angles[0]), 6),
            "l2cs_pitch_rad": round(float(angles[1]), 6),
            "l2cs_yaw_deg": round(float(np.degrees(angles[0])), 3),
            "l2cs_pitch_deg": round(float(np.degrees(angles[1])), 3),
            "raw_point": {"x": round(float(raw[0]), 2), "y": round(float(raw[1]), 2)},
            "clipped": bool(
                abs(float(raw[0]) - clipped[0]) > 1e-6
                or abs(float(raw[1]) - clipped[1]) > 1e-6
            ),
            "out_of_calibration_range": bool(out_indices),
            "out_of_range_feature_indices": out_indices,
            "range_excess": round(float(np.max(excess)), 4),
        }

    @staticmethod
    def _basis(normalized_features: np.ndarray, degree: int) -> np.ndarray:
        values = np.asarray(normalized_features, dtype=np.float64)
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values.reshape(1, -1)

        yaw = values[:, 0]
        pitch = values[:, 1]
        columns = [yaw, pitch]
        if int(degree) >= 2:
            columns.extend([yaw * pitch, yaw * yaw, pitch * pitch])
        columns.append(np.ones(values.shape[0], dtype=np.float64))
        basis = np.column_stack(columns)
        return basis[0] if one_dimensional else basis


class L2CSLocalKnnGazeModel:
    """Local L2CS yaw/pitch interpolation over dense calibration anchors."""

    model_type = "l2cs_local_knn_gaze_v1"
    feature_count = 2

    def __init__(
        self,
        anchor_features: np.ndarray,
        targets: np.ndarray,
        screen_size: tuple[int, int],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
        neighbor_count: int,
        range_margin: float,
        signal_stats: dict[str, float],
        validation_error_px: float | None = None,
        training_error_px: float | None = None,
        sample_count: int = 0,
        point_errors_px: list[float] | None = None,
        created_at: float | None = None,
    ) -> None:
        self.anchor_features = np.asarray(anchor_features, dtype=np.float64)
        self.targets = np.asarray(targets, dtype=np.float64)
        self.screen_size = screen_size
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.feature_min = np.asarray(feature_min, dtype=np.float64)
        self.feature_max = np.asarray(feature_max, dtype=np.float64)
        self.neighbor_count = max(1, int(neighbor_count))
        self.range_margin = max(0.0, float(range_margin))
        self.signal_stats = signal_stats
        self.x_signal_range = float(signal_stats.get("yaw_range_rad", 0.0))
        self.y_signal_range = float(signal_stats.get("pitch_range_rad", 0.0))
        self.validation_error_px = validation_error_px
        self.training_error_px = training_error_px
        self.sample_count = sample_count
        self.point_errors_px = point_errors_px or []
        self.created_at = created_at or time.time()

    @classmethod
    def fit(
        cls,
        feature_rows: list[list[float]],
        targets: list[tuple[float, float]],
        screen_size: tuple[int, int],
        neighbor_count: int,
        range_margin: float,
        validation_error_px: float | None = None,
    ) -> "L2CSLocalKnnGazeModel":
        if len(feature_rows) < 6:
            raise ValueError("At least 6 L2CS angle samples are required.")
        if len(feature_rows) != len(targets):
            raise ValueError("Feature and target sample counts must match.")

        features = np.asarray(feature_rows, dtype=np.float64)[:, : cls.feature_count]
        target_values = np.asarray(targets, dtype=np.float64)
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
        signal_stats = {
            "yaw_range_rad": float(np.max(features[:, 0]) - np.min(features[:, 0])),
            "pitch_range_rad": float(np.max(features[:, 1]) - np.min(features[:, 1])),
            "yaw_range_deg": float(
                np.degrees(np.max(features[:, 0]) - np.min(features[:, 0]))
            ),
            "pitch_range_deg": float(
                np.degrees(np.max(features[:, 1]) - np.min(features[:, 1]))
            ),
            "yaw_min_rad": float(np.min(features[:, 0])),
            "yaw_max_rad": float(np.max(features[:, 0])),
            "pitch_min_rad": float(np.min(features[:, 1])),
            "pitch_max_rad": float(np.max(features[:, 1])),
        }
        model = cls(
            anchor_features=features,
            targets=target_values,
            screen_size=screen_size,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            feature_min=np.min(features, axis=0),
            feature_max=np.max(features, axis=0),
            neighbor_count=neighbor_count,
            range_margin=range_margin,
            signal_stats=signal_stats,
            validation_error_px=validation_error_px,
            sample_count=len(feature_rows),
        )
        predictions = np.asarray(
            [model._predict_selected(features[index], exclude_index=index)[0] for index in range(len(features))]
        )
        errors = np.linalg.norm(predictions - target_values, axis=1)
        model.training_error_px = float(np.mean(errors))
        return model

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "L2CSLocalKnnGazeModel":
        return cls(
            anchor_features=np.asarray(payload["anchor_features"], dtype=np.float64),
            targets=np.asarray(payload["targets"], dtype=np.float64),
            screen_size=(int(payload["screen_width"]), int(payload["screen_height"])),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            feature_min=np.asarray(payload["feature_min"], dtype=np.float64),
            feature_max=np.asarray(payload["feature_max"], dtype=np.float64),
            neighbor_count=int(payload.get("neighbor_count", 8)),
            range_margin=float(payload.get("range_margin", 0.08)),
            signal_stats={
                key: float(value) for key, value in payload.get("signal_stats", {}).items()
            },
            validation_error_px=payload.get("validation_error_px"),
            training_error_px=payload.get("training_error_px"),
            sample_count=int(payload.get("sample_count", 0)),
            point_errors_px=_optional_float_list(payload.get("point_errors_px", [])),
            created_at=payload.get("created_at"),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "created_at": self.created_at,
            "feature_count": self.feature_count,
            "screen_width": self.screen_size[0],
            "screen_height": self.screen_size[1],
            "sample_count": self.sample_count,
            "training_error_px": self.training_error_px,
            "validation_error_px": self.validation_error_px,
            "point_errors_px": self.point_errors_px,
            "neighbor_count": self.neighbor_count,
            "range_margin": self.range_margin,
            "signal_stats": self.signal_stats,
            "anchor_features": self.anchor_features.tolist(),
            "targets": self.targets.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
        }
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def predict(self, features: list[float] | np.ndarray) -> tuple[float, float]:
        point, _debug = self.predict_with_diagnostics(features)
        return point

    def predict_with_diagnostics(
        self,
        features: list[float] | np.ndarray,
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.shape[0] < self.feature_count:
            raise ValueError(f"Expected at least 2 L2CS angle features, got {feature_array.shape[0]}.")

        selected = feature_array[: self.feature_count]
        point, distances = self._predict_selected(selected)
        debug = self._diagnostics(selected, distances)
        width, height = self.screen_size
        clipped = (
            float(np.clip(point[0], 0, max(0, width - 1))),
            float(np.clip(point[1], 0, max(0, height - 1))),
        )
        return clipped, debug

    def _predict_selected(
        self,
        selected: np.ndarray,
        exclude_index: int | None = None,
    ) -> tuple[tuple[float, float], np.ndarray]:
        normalized_delta = (self.anchor_features - selected) / self.feature_scale
        distances = np.linalg.norm(normalized_delta, axis=1)
        if exclude_index is not None and len(distances) > 1:
            distances = distances.copy()
            distances[exclude_index] = np.inf

        order = np.argsort(distances)
        finite_order = [index for index in order if np.isfinite(distances[index])]
        if not finite_order:
            center = np.mean(self.targets, axis=0)
            return (float(center[0]), float(center[1])), distances

        best = finite_order[0]
        if distances[best] < 1e-9:
            target = self.targets[best]
            return (float(target[0]), float(target[1])), distances

        chosen = finite_order[: min(self.neighbor_count, len(finite_order))]
        chosen_distances = distances[chosen]
        weights = 1.0 / np.square(chosen_distances + 1e-6)
        weights = weights / np.sum(weights)
        point = weights @ self.targets[chosen]
        return (float(point[0]), float(point[1])), distances

    def _diagnostics(self, selected: np.ndarray, distances: np.ndarray) -> dict[str, Any]:
        feature_range = np.maximum(self.feature_max - self.feature_min, 1e-6)
        lower = self.feature_min - feature_range * self.range_margin
        upper = self.feature_max + feature_range * self.range_margin
        below = np.maximum(lower - selected, 0.0) / feature_range
        above = np.maximum(selected - upper, 0.0) / feature_range
        excess = below + above
        out_indices = [index for index, value in enumerate(excess) if float(value) > 0.0]
        finite_distances = distances[np.isfinite(distances)]
        nearest = float(np.min(finite_distances)) if finite_distances.size else None
        ordered = np.sort(finite_distances)[: self.neighbor_count]
        return {
            "model_kind": self.model_type,
            "selected_features": {
                "l2cs_yaw_rad": round(float(selected[0]), 6),
                "l2cs_pitch_rad": round(float(selected[1]), 6),
                "l2cs_yaw_deg": round(float(np.degrees(selected[0])), 3),
                "l2cs_pitch_deg": round(float(np.degrees(selected[1])), 3),
            },
            "nearest_feature_distance": None if nearest is None else round(nearest, 4),
            "nearest_distances": [round(float(value), 4) for value in ordered],
            "out_of_calibration_range": bool(out_indices),
            "out_of_range_feature_indices": out_indices,
            "range_excess": round(float(np.max(excess)), 4),
        }


class EmaPointSmoother:
    """Median + step-limited exponential smoother for screen-space gaze points."""

    def __init__(
        self,
        alpha: float,
        median_window: int = 1,
        max_step_px: float | None = None,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_step_px = None if max_step_px is None else float(max_step_px)
        self._raw_points: deque[tuple[float, float]] = deque(maxlen=max(1, int(median_window)))
        self._point: tuple[float, float] | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        self._raw_points.append(point)
        point = self._median_point()

        if self._point is None:
            self._point = point
            return point

        prev_x, prev_y = self._point
        x, y = point
        if self.max_step_px is not None:
            dx = x - prev_x
            dy = y - prev_y
            distance = float(np.hypot(dx, dy))
            if distance > self.max_step_px and distance > 1e-6:
                scale = self.max_step_px / distance
                x = prev_x + dx * scale
                y = prev_y + dy * scale

        smoothed = (
            self.alpha * x + (1.0 - self.alpha) * prev_x,
            self.alpha * y + (1.0 - self.alpha) * prev_y,
        )
        self._point = smoothed
        return smoothed

    def reset(self) -> None:
        self._point = None
        self._raw_points.clear()

    def _median_point(self) -> tuple[float, float]:
        points = np.asarray(self._raw_points, dtype=np.float64)
        median = np.median(points, axis=0)
        return float(median[0]), float(median[1])
