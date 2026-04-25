"""OpenCV drawing helpers for the live-only debug preview."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import numpy as np


Color = tuple[int, int, int]
Point = tuple[int, int]


def screen_to_preview_point(
    point: tuple[float, float],
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> Point:
    """Map a full-screen pixel point into preview-window pixels."""
    screen_width, screen_height = screen_size
    preview_width, preview_height = preview_size
    x = int(round(point[0] * preview_width / max(1, screen_width)))
    y = int(round(point[1] * preview_height / max(1, screen_height)))
    return clamp_point((x, y), preview_size)


def preview_to_screen_point(
    point: tuple[int, int],
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[float, float]:
    """Map a preview-window pixel point into full-screen pixel coordinates."""
    screen_width, screen_height = screen_size
    preview_width, preview_height = preview_size
    x = float(point[0] * screen_width / max(1, preview_width))
    y = float(point[1] * screen_height / max(1, preview_height))
    return x, y


def screen_length_to_preview_length(
    length: float,
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> int:
    """Map a screen-space length to the current preview scale."""
    screen_width, screen_height = screen_size
    preview_width, preview_height = preview_size
    scale = min(
        preview_width / max(1, screen_width),
        preview_height / max(1, screen_height),
    )
    return max(1, int(round(float(length) * scale)))


def screen_bbox_to_preview_bbox(
    bbox: Sequence[float],
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map [x1, y1, x2, y2] from screen pixels to preview pixels."""
    x1, y1 = screen_to_preview_point(
        (float(bbox[0]), float(bbox[1])), preview_size, screen_size
    )
    x2, y2 = screen_to_preview_point(
        (float(bbox[2]), float(bbox[3])), preview_size, screen_size
    )
    return x1, y1, x2, y2


def clamp_point(point: tuple[int, int], preview_size: tuple[int, int]) -> Point:
    """Clamp a preview pixel point so drawing stays inside the frame."""
    width, height = preview_size
    x = min(max(0, int(point[0])), max(0, width - 1))
    y = min(max(0, int(point[1])), max(0, height - 1))
    return x, y


def blank_preview(preview_size: tuple[int, int]) -> np.ndarray:
    """Create a neutral preview canvas representing the primary monitor."""
    width, height = preview_size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (28, 30, 34)

    grid_color = (45, 49, 55)
    step = max(80, width // 8)
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)

    return frame


def draw_soft_gaze_blob(
    frame: np.ndarray,
    center: tuple[int, int] | None,
    radius: int,
    alpha: float,
    color: Color = (0, 220, 255),
) -> None:
    """Draw a translucent gaze blob in-place."""
    if center is None:
        return

    overlay = frame.copy()
    center = clamp_point(center, (frame.shape[1], frame.shape[0]))

    rings = ((1.0, alpha * 0.30), (0.62, alpha * 0.55), (0.26, alpha))
    for fraction, ring_alpha in rings:
        ring_radius = max(2, int(radius * fraction))
        cv2.circle(overlay, center, ring_radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, ring_alpha, frame, 1.0 - ring_alpha, 0, dst=frame)
        overlay = frame.copy()

    cv2.circle(frame, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, center, 8, color, 1, cv2.LINE_AA)


def draw_gaze_hit_circle(
    frame: np.ndarray,
    center: tuple[int, int] | None,
    radius: int,
    color: Color = (0, 220, 255),
) -> None:
    """Draw the actual hit radius used by torso intersection logic."""
    if center is None or radius <= 0:
        return
    center = clamp_point(center, (frame.shape[1], frame.shape[0]))
    cv2.circle(frame, center, int(radius), color, 2, cv2.LINE_AA)


def draw_gaze_trail(
    frame: np.ndarray,
    trail: Sequence[tuple[int, int]],
    color: Color = (0, 180, 255),
) -> None:
    """Draw a short fading in-memory trail."""
    if len(trail) < 2:
        return

    total = len(trail)
    for index in range(1, total):
        fade = index / total
        thickness = max(1, int(1 + fade * 5))
        point_a = clamp_point(trail[index - 1], (frame.shape[1], frame.shape[0]))
        point_b = clamp_point(trail[index], (frame.shape[1], frame.shape[0]))
        segment_color = tuple(int(channel * fade) for channel in color)
        cv2.line(frame, point_a, point_b, segment_color, thickness, cv2.LINE_AA)


def draw_alert_flash(
    frame: np.ndarray,
    alert_state: dict[str, Any] | None,
    color: Color = (0, 80, 255),
    fill_alpha: float = 0.16,
) -> None:
    """Draw a controlled visual-only flash when an alert is active."""
    if not alert_state or not alert_state.get("active"):
        return

    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), color, -1)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, dst=frame)

    border = max(6, min(width, height) // 60)
    cv2.rectangle(
        frame,
        (border // 2, border // 2),
        (width - border // 2 - 1, height - border // 2 - 1),
        color,
        border,
        cv2.LINE_AA,
    )

    label = str(alert_state.get("active_label") or alert_state.get("matched_label") or "target")
    _draw_label(frame, f"ALERT {label}", (border + 8, border + 28), color)


def draw_detection_boxes(
    frame: np.ndarray,
    detected_objects: Iterable[dict[str, Any]],
    matched_objects: Iterable[dict[str, Any]],
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
    target_labels: set[str] | frozenset[str],
) -> None:
    """Draw detections and highlight matched target boxes."""
    match_keys = {
        key
        for obj in matched_objects
        if isinstance(obj, dict)
        for key in [_object_key(obj)]
        if key is not None
    }

    for obj in detected_objects:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        if not _valid_bbox_shape(bbox):
            continue

        x1, y1, x2, y2 = screen_bbox_to_preview_bbox(bbox, preview_size, screen_size)
        label = str(obj.get("label", "unknown"))
        confidence = _safe_float(obj.get("confidence", 0.0))
        is_target = label in target_labels
        is_hit = _object_key(obj) in match_keys

        color = (110, 110, 110)
        if is_target:
            color = (0, 210, 255)
        if is_hit:
            color = (0, 255, 90)

        _draw_mask_polygons(
            frame,
            obj.get("mask_polygons"),
            preview_size,
            screen_size,
            color,
            fill_alpha=0.20 if is_hit else 0.10,
        )

        thickness = 3 if is_hit else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label_text = f"{label} {confidence:.2f}"
        if is_hit:
            label_text = f"HIT {label} {confidence:.2f}"
        _draw_label(frame, label_text, (x1, max(0, y1 - 8)), color)


def draw_status_panel(frame: np.ndarray, lines: Sequence[str]) -> None:
    """Draw compact status text in the top-left corner."""
    if not lines:
        return

    x = 12
    y = 16
    line_height = 22
    panel_width = min(frame.shape[1] - 20, 620)
    panel_height = 18 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (12, 14, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 238, 242),
            1,
            cv2.LINE_AA,
        )


def _draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: Color) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = min(max(0, x), max(0, frame.shape[1] - text_width - 8))
    y = min(max(text_height + 6, y), max(text_height + 6, frame.shape[0] - 4))
    cv2.rectangle(
        frame,
        (x, y - text_height - 6),
        (x + text_width + 8, y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y),
        font,
        scale,
        (10, 12, 16),
        thickness,
        cv2.LINE_AA,
    )


def _draw_mask_polygons(
    frame: np.ndarray,
    mask_polygons: Any,
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
    color: Color,
    fill_alpha: float,
) -> None:
    preview_polygons = _screen_mask_polygons_to_preview(
        mask_polygons,
        preview_size,
        screen_size,
    )
    if not preview_polygons:
        return

    overlay = frame.copy()
    cv2.fillPoly(overlay, preview_polygons, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, dst=frame)
    for polygon in preview_polygons:
        cv2.polylines(frame, [polygon], True, color, 1, cv2.LINE_AA)


def _screen_mask_polygons_to_preview(
    mask_polygons: Any,
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> list[np.ndarray]:
    if not isinstance(mask_polygons, Sequence) or isinstance(mask_polygons, (str, bytes)):
        return []

    converted: list[np.ndarray] = []
    for polygon in mask_polygons:
        if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
            continue
        points: list[tuple[int, int]] = []
        for point in polygon:
            if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
                continue
            if len(point) < 2:
                continue
            try:
                screen_point = (float(point[0]), float(point[1]))
            except (TypeError, ValueError):
                continue
            points.append(screen_to_preview_point(screen_point, preview_size, screen_size))
        if len(points) >= 3:
            converted.append(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2))
    return converted


def _object_key(obj: dict[str, Any]) -> tuple[str, tuple[float, float, float, float], float] | None:
    bbox = obj.get("bbox")
    if not _valid_bbox_shape(bbox):
        return None
    return (
        str(obj.get("label", "")),
        tuple(round(float(value), 3) for value in bbox),
        round(_safe_float(obj.get("confidence", 0.0)), 3),
    )


def _valid_bbox_shape(bbox: Any) -> bool:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        return False
    if len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return x2 >= x1 and y2 >= y1


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
