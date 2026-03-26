"""
Tests for gaze logic in src/processors/gaze_udf.py (no PySpark).
"""

import sys
from pathlib import Path

# Repo root → src/processors on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "processors"))

from gaze_udf import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    SENSITIVE_LABELS,
    _is_gaze_in_sensitive_bbox,
    _matched_confidence,
    _matched_label,
)

HEAD_OBJ = {
    "label": "head",
    "bbox": [300.0, 100.0, 500.0, 300.0],
    "confidence": 0.92,
}


def test_gaze_inside_bbox_returns_true():
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, [HEAD_OBJ]) is True


def test_gaze_outside_bbox_returns_false():
    assert _is_gaze_in_sensitive_bbox(100.0, 50.0, [HEAD_OBJ]) is False


def test_gaze_on_bbox_edge_returns_true():
    assert _is_gaze_in_sensitive_bbox(300.0, 100.0, [HEAD_OBJ]) is True


def test_non_sensitive_label_returns_false():
    obj = {"label": "leg", "bbox": [300.0, 100.0, 500.0, 300.0], "confidence": 0.92}
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, [obj]) is False


def test_low_confidence_returns_false():
    obj = {**HEAD_OBJ, "confidence": 0.1}
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, [obj]) is False


def test_none_objects_returns_false():
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, None) is False


def test_empty_objects_returns_false():
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, []) is False


def test_multiple_objects_hit_second():
    floor_obj = {
        "label": "floor",
        "bbox": [0.0, 400.0, 640.0, 480.0],
        "confidence": 0.88,
    }
    assert _is_gaze_in_sensitive_bbox(400.0, 200.0, [floor_obj, HEAD_OBJ]) is True


def test_matched_label_returns_label():
    assert _matched_label(400.0, 200.0, [HEAD_OBJ]) == "head"


def test_matched_label_returns_none_on_miss():
    assert _matched_label(100.0, 50.0, [HEAD_OBJ]) is None


def test_matched_confidence_returns_value():
    assert abs(_matched_confidence(400.0, 200.0, [HEAD_OBJ]) - 0.92) < 0.001


def test_matched_confidence_returns_zero_on_miss():
    assert _matched_confidence(100.0, 50.0, [HEAD_OBJ]) == 0.0


def test_sensitive_labels_and_threshold_exported():
    assert "head" in SENSITIVE_LABELS
    assert CONFIDENCE_THRESHOLD == 0.5
