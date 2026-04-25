"""YOLO detector wrapper for live screen frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.torch_device import resolve_torch_device


def clip_bbox_to_frame(
    bbox: Sequence[float],
    frame_size: tuple[int, int],
) -> list[float] | None:
    """Clip an xyxy bbox to a frame and return screen-pixel floats."""
    if len(bbox) != 4:
        return None

    width, height = frame_size
    if width <= 0 or height <= 0:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    x1 = min(max(0.0, x1), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    x2 = min(max(0.0, x2), float(width - 1))
    y2 = min(max(0.0, y2), float(height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def scale_bbox(
    bbox: Sequence[float],
    frame_size: tuple[int, int],
    scale: float,
) -> list[float] | None:
    """Scale a bbox around its center and clip to the frame."""
    clipped = clip_bbox_to_frame(bbox, frame_size)
    if clipped is None:
        return None
    try:
        scale_value = max(0.01, float(scale))
    except (TypeError, ValueError):
        scale_value = 1.0
    if abs(scale_value - 1.0) < 1e-9:
        return clipped

    x1, y1, x2, y2 = clipped
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * scale_value / 2.0
    half_height = (y2 - y1) * scale_value / 2.0
    return clip_bbox_to_frame(
        [center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height],
        frame_size,
    )


def make_detection(
    label: str,
    bbox: Sequence[float],
    confidence: float,
    frame_size: tuple[int, int],
    mask_polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    geometry_scale: float = 1.0,
) -> dict[str, Any] | None:
    try:
        scale_value = max(0.01, float(geometry_scale))
    except (TypeError, ValueError):
        scale_value = 1.0

    clipped = scale_bbox(bbox, frame_size, scale_value)
    if clipped is None:
        return None
    detection: dict[str, Any] = {
        "label": str(label),
        "bbox": clipped,
        "confidence": round(float(confidence), 4),
    }
    valid_mask_polygons = normalize_mask_polygons(
        mask_polygons,
        frame_size,
        scale=scale_value,
    )
    if valid_mask_polygons:
        detection["mask_polygons"] = valid_mask_polygons
        detection["has_mask"] = True
        detection["mask_point_count"] = sum(
            len(polygon) for polygon in valid_mask_polygons
        )
    if abs(scale_value - 1.0) > 1e-9:
        detection["geometry_scale"] = round(scale_value, 3)
    return detection


def label_from_names(names: Mapping[int, str] | Sequence[str], class_index: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_index, class_index))
    if 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


class YoloScreenDetector:
    """Load a local YOLO model and run object detection on screen frames."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        image_size: int,
        device: str,
        geometry_scale: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise RuntimeError(
                f"YOLO model is missing: {self.model_path}. "
                "Run download_models.py and make sure the trained body-region model is present."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError(
                "Missing dependency: ultralytics. Install requirements.txt first."
            ) from exc

        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = resolve_torch_device(device)
        self.geometry_scale = max(0.01, float(geometry_scale))
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        if frame_bgr is None or frame_bgr.ndim != 3:
            return []

        frame_height, frame_width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        return detections_from_yolo_result(
            results[0],
            names=self.names,
            frame_size=(frame_width, frame_height),
            geometry_scale=self.geometry_scale,
        )


def detections_from_yolo_result(
    result: Any,
    names: Mapping[int, str] | Sequence[str],
    frame_size: tuple[int, int],
    geometry_scale: float = 1.0,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    mask_polygons_by_box = mask_polygons_from_yolo_result(result, frame_size)

    detections: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        xyxy_values = _tensor_values(box.xyxy[0])
        confidence_values = _tensor_values(box.conf)
        class_values = _tensor_values(box.cls)
        if not xyxy_values or not confidence_values or not class_values:
            continue

        class_index = int(class_values[0])
        mask_polygons = []
        if index < len(mask_polygons_by_box):
            mask_polygons = mask_polygons_by_box[index]
        detection = make_detection(
            label=label_from_names(names, class_index),
            bbox=xyxy_values,
            confidence=float(confidence_values[0]),
            frame_size=frame_size,
            mask_polygons=mask_polygons,
            geometry_scale=geometry_scale,
        )
        if detection is not None:
            detections.append(detection)
    return detections


def mask_polygons_from_yolo_result(
    result: Any,
    frame_size: tuple[int, int],
) -> list[list[list[list[float]]]]:
    masks = getattr(result, "masks", None)
    raw_polygons = getattr(masks, "xy", None) if masks is not None else None
    if raw_polygons is None:
        return []

    polygons_by_box: list[list[list[list[float]]]] = []
    for raw_polygon in raw_polygons:
        normalized = normalize_mask_polygons([raw_polygon], frame_size)
        polygons_by_box.append(normalized)
    return polygons_by_box


def normalize_mask_polygons(
    mask_polygons: Sequence[Sequence[Sequence[float]]] | None,
    frame_size: tuple[int, int],
    scale: float = 1.0,
) -> list[list[list[float]]]:
    if not mask_polygons:
        return []

    width, height = frame_size
    if width <= 0 or height <= 0:
        return []

    try:
        scale_value = max(0.01, float(scale))
    except (TypeError, ValueError):
        scale_value = 1.0

    normalized: list[list[list[float]]] = []
    for polygon in mask_polygons:
        try:
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if len(points) < 3:
            continue
        if abs(scale_value - 1.0) > 1e-9:
            finite_points = points[np.isfinite(points).all(axis=1)]
            if len(finite_points) >= 3:
                center = finite_points.mean(axis=0)
                points = center + (points - center) * scale_value

        clean_polygon: list[list[float]] = []
        previous: tuple[float, float] | None = None
        for x_value, y_value in points:
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            x = min(max(0.0, float(x_value)), float(width - 1))
            y = min(max(0.0, float(y_value)), float(height - 1))
            rounded = (round(x, 1), round(y, 1))
            if previous == rounded:
                continue
            clean_polygon.append([rounded[0], rounded[1]])
            previous = rounded

        if len(clean_polygon) >= 3:
            normalized.append(clean_polygon)
    return normalized


def _tensor_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        raw = value.tolist()
    else:
        raw = value
    if isinstance(raw, list):
        return [float(item) for item in raw]
    return [float(raw)]
