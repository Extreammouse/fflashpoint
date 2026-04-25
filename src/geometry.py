"""Pure Python geometry helpers for screen-space gaze and detector boxes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = list[Point]


def _coerce_bbox(bbox: Any) -> BBox | None:
    """Return a valid [x1, y1, x2, y2] bbox as floats, or None."""
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        return None
    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    if x2 < x1 or y2 < y1:
        return None

    return x1, y1, x2, y2


def _coerce_point(gaze_point: Any) -> Point | None:
    """Accept either {"x": x, "y": y} or a two-item point sequence."""
    if gaze_point is None:
        return None

    try:
        if isinstance(gaze_point, Mapping):
            return float(gaze_point["x"]), float(gaze_point["y"])
        if not isinstance(gaze_point, (str, bytes)) and isinstance(
            gaze_point, Sequence
        ):
            if len(gaze_point) < 2:
                return None
            return float(gaze_point[0]), float(gaze_point[1])
    except (KeyError, TypeError, ValueError):
        return None

    return None


def point_in_bbox(x: float, y: float, bbox: Any) -> bool:
    """Return True when point (x, y) is inside or on bbox edges.

    Coordinates are full primary-monitor screen pixels. Bbox format is
    [x1, y1, x2, y2] with top-left and bottom-right corners.
    """
    valid_bbox = _coerce_bbox(bbox)
    if valid_bbox is None:
        return False

    try:
        px = float(x)
        py = float(y)
    except (TypeError, ValueError):
        return False

    x1, y1, x2, y2 = valid_bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _coerce_polygon(polygon: Any) -> Polygon | None:
    if isinstance(polygon, (str, bytes)) or not isinstance(polygon, Sequence):
        return None
    if len(polygon) < 3:
        return None

    points: Polygon = []
    for point in polygon:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
            return None
        if len(point) < 2:
            return None
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
    return points


def point_in_polygon(x: float, y: float, polygon: Any) -> bool:
    """Return True when point (x, y) is inside or on a polygon edge."""
    valid_polygon = _coerce_polygon(polygon)
    if valid_polygon is None:
        return False

    try:
        px = float(x)
        py = float(y)
    except (TypeError, ValueError):
        return False

    inside = False
    point_count = len(valid_polygon)
    previous_x, previous_y = valid_polygon[-1]
    for current_x, current_y in valid_polygon:
        if _point_on_segment(px, py, previous_x, previous_y, current_x, current_y):
            return True
        crosses_y = (current_y > py) != (previous_y > py)
        if crosses_y:
            x_intersection = (previous_x - current_x) * (py - current_y) / (
                previous_y - current_y
            ) + current_x
            if px <= x_intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y

    return inside and point_count >= 3


def point_in_mask_polygons(x: float, y: float, mask_polygons: Any) -> bool:
    """Return True when a point intersects any valid segmentation polygon."""
    if isinstance(mask_polygons, (str, bytes)) or not isinstance(mask_polygons, Sequence):
        return False
    return any(point_in_polygon(x, y, polygon) for polygon in mask_polygons)


def circle_intersects_bbox(x: float, y: float, radius: float, bbox: Any) -> bool:
    """Return True when a point-radius circle intersects a bbox."""
    valid_bbox = _coerce_bbox(bbox)
    if valid_bbox is None:
        return False
    try:
        px = float(x)
        py = float(y)
        circle_radius = max(0.0, float(radius))
    except (TypeError, ValueError):
        return False

    x1, y1, x2, y2 = valid_bbox
    nearest_x = min(max(px, x1), x2)
    nearest_y = min(max(py, y1), y2)
    return _squared_distance(px, py, nearest_x, nearest_y) <= circle_radius * circle_radius


def circle_intersects_polygon(x: float, y: float, radius: float, polygon: Any) -> bool:
    """Return True when a point-radius circle intersects a polygon."""
    valid_polygon = _coerce_polygon(polygon)
    if valid_polygon is None:
        return False
    try:
        px = float(x)
        py = float(y)
        circle_radius = max(0.0, float(radius))
    except (TypeError, ValueError):
        return False

    if point_in_polygon(px, py, valid_polygon):
        return True

    radius_squared = circle_radius * circle_radius
    previous_x, previous_y = valid_polygon[-1]
    for current_x, current_y in valid_polygon:
        if (
            _squared_distance_to_segment(
                px,
                py,
                previous_x,
                previous_y,
                current_x,
                current_y,
            )
            <= radius_squared
        ):
            return True
        previous_x, previous_y = current_x, current_y
    return False


def circle_intersects_mask_polygons(
    x: float,
    y: float,
    radius: float,
    mask_polygons: Any,
) -> bool:
    """Return True when a point-radius circle intersects any mask polygon."""
    if isinstance(mask_polygons, (str, bytes)) or not isinstance(mask_polygons, Sequence):
        return False
    return any(circle_intersects_polygon(x, y, radius, polygon) for polygon in mask_polygons)


def detection_contains_point(x: float, y: float, detection: Mapping[str, Any]) -> tuple[bool, str]:
    """Hit-test one detection as an exact point."""
    return detection_intersects_gaze(x, y, 0.0, detection)


def detection_intersects_gaze(
    x: float,
    y: float,
    radius: float,
    detection: Mapping[str, Any],
) -> tuple[bool, str]:
    """Hit-test one detection, preferring mask polygons and falling back to bbox."""
    mask_polygons = detection.get("mask_polygons")
    radius_value = _safe_radius(radius)
    if circle_intersects_mask_polygons(x, y, radius_value, mask_polygons):
        return True, "mask" if radius_value == 0 else "mask_radius"
    if mask_polygons:
        return False, "mask"
    if circle_intersects_bbox(x, y, radius_value, detection.get("bbox")):
        return True, "bbox" if radius_value == 0 else "bbox_radius"
    return False, "bbox"


def _point_on_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tolerance: float = 1e-9,
) -> bool:
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > tolerance:
        return False

    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= tolerance


def _safe_radius(radius: Any) -> float:
    try:
        return max(0.0, float(radius))
    except (TypeError, ValueError):
        return 0.0


def _squared_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def _squared_distance_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return _squared_distance(px, py, x1, y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_squared
    t = min(max(0.0, t), 1.0)
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return _squared_distance(px, py, nearest_x, nearest_y)


def find_gaze_hits(
    gaze_point: Any,
    detected_objects: Any,
    target_labels: set[str] | frozenset[str],
    confidence_threshold: float,
    gaze_radius_px: float = 0.0,
) -> list[dict[str, Any]]:
    """Return target detections whose geometry intersects the gaze point/radius."""
    point = _coerce_point(gaze_point)
    if point is None or detected_objects is None:
        return []

    if not isinstance(detected_objects, Sequence) or isinstance(
        detected_objects, (str, bytes)
    ):
        return []

    target_label_set = set(target_labels)
    hits: list[dict[str, Any]] = []
    gaze_x, gaze_y = point

    for obj in detected_objects:
        if not isinstance(obj, Mapping):
            continue

        label = obj.get("label")
        if label not in target_label_set:
            continue

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        if confidence < confidence_threshold:
            continue

        contains_point, hit_region = detection_intersects_gaze(
            gaze_x,
            gaze_y,
            gaze_radius_px,
            obj,
        )
        if not contains_point:
            continue

        hit = dict(obj)
        hit["hit_region"] = hit_region
        hit["gaze_radius_px"] = round(_safe_radius(gaze_radius_px), 1)
        hits.append(hit)

    return hits
