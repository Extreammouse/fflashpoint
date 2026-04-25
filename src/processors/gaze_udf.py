"""PySpark UDFs for spatial gaze-hit detection in the FLASH-POINT pipeline.

Can also be imported as plain Python for unit testing — no Spark context required.
"""

from __future__ import annotations

SENSITIVE_LABELS: set[str] = {"upper_torso"}
CONFIDENCE_THRESHOLD: float = 0.5


def _is_gaze_in_sensitive_bbox(
    gaze_x: float,
    gaze_y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float = 0.0,
) -> bool:
    """Return True when the gaze circle intersects a bounding box."""
    try:
        gx, gy = float(gaze_x), float(gaze_y)
        bx1, by1, bx2, by2 = float(x1), float(y1), float(x2), float(y2)
        r = max(0.0, float(radius))
    except (TypeError, ValueError):
        return False

    if bx2 < bx1 or by2 < by1:
        return False

    nearest_x = min(max(gx, bx1), bx2)
    nearest_y = min(max(gy, by1), by2)
    dist_sq = (gx - nearest_x) ** 2 + (gy - nearest_y) ** 2
    return dist_sq <= r * r if r > 0 else (bx1 <= gx <= bx2 and by1 <= gy <= by2)


def _matched_label(
    detections: list[dict],
    gaze_x: float,
    gaze_y: float,
    radius: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sensitive_labels: set[str] = SENSITIVE_LABELS,
) -> str | None:
    """Return the label of the first sensitive detection hit, or None."""
    for det in detections or []:
        label = det.get("label")
        if label not in sensitive_labels:
            continue
        try:
            conf = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < confidence_threshold:
            continue
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
        if _is_gaze_in_sensitive_bbox(gaze_x, gaze_y, *bbox[:4], radius=radius):
            return label
    return None


def _matched_confidence(
    detections: list[dict],
    gaze_x: float,
    gaze_y: float,
    radius: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sensitive_labels: set[str] = SENSITIVE_LABELS,
) -> float:
    """Return the confidence of the first sensitive detection hit, or 0.0."""
    for det in detections or []:
        label = det.get("label")
        if label not in sensitive_labels:
            continue
        try:
            conf = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < confidence_threshold:
            continue
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
        if _is_gaze_in_sensitive_bbox(gaze_x, gaze_y, *bbox[:4], radius=radius):
            return conf
    return 0.0


def _register_udfs(spark):
    """Register PySpark UDFs on a SparkSession. Called from the Spark job."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, StringType, FloatType

    def _hit_bool(gaze_x, gaze_y, x1, y1, x2, y2, label, confidence):
        if label not in SENSITIVE_LABELS:
            return False
        try:
            if float(confidence) < CONFIDENCE_THRESHOLD:
                return False
        except (TypeError, ValueError):
            return False
        return _is_gaze_in_sensitive_bbox(gaze_x, gaze_y, x1, y1, x2, y2, radius=150.0)

    spark.udf.register("gaze_hit", _hit_bool, BooleanType())
    return spark
