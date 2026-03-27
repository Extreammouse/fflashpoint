"""
Pure-Python gaze / bbox logic used by spark_analytics UDFs.
No PySpark import — safe to unit test without Spark installed.
"""

SENSITIVE_LABELS = {"head"}
CONFIDENCE_THRESHOLD = 0.5
 


def _is_gaze_in_sensitive_bbox(gaze_x, gaze_y, detected_objects):
    """
    Returns True if (gaze_x, gaze_y) falls inside the bounding box of ANY
    detected object whose label is in SENSITIVE_LABELS and whose confidence
    meets CONFIDENCE_THRESHOLD.

    bbox format: [x1, y1, x2, y2]  (top-left → bottom-right, pixel coords)
    """
    if detected_objects is None:
        
        return False

    for obj in detected_objects:
        label = obj["label"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]

        if label not in SENSITIVE_LABELS:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return True

    return False


def _matched_label(gaze_x, gaze_y, detected_objects):
    """Returns the label of the first matched sensitive object, or None."""
    if detected_objects is None:
        return None

    for obj in detected_objects:
        label = obj["label"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]

        if label not in SENSITIVE_LABELS:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return label

    return None


def _matched_confidence(gaze_x, gaze_y, detected_objects):
    """Returns the confidence of the first matched sensitive object, or 0.0."""
    if detected_objects is None:
        return 0.0

    for obj in detected_objects:
        label = obj["label"]
        confidence = obj["confidence"]
        bbox = obj["bbox"]

        if label not in SENSITIVE_LABELS:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return float(confidence)

    return 0.0
