"""Unit tests for gaze_udf spatial hit detection."""

import pytest
from src.processors.gaze_udf import (
    _is_gaze_in_sensitive_bbox,
    _matched_label,
    _matched_confidence,
    SENSITIVE_LABELS,
    CONFIDENCE_THRESHOLD,
)


# ── _is_gaze_in_sensitive_bbox ─────────────────────────────────────────────────

def test_point_inside_bbox():
    assert _is_gaze_in_sensitive_bbox(200, 300, 100, 200, 400, 600) is True


def test_point_on_bbox_edge():
    assert _is_gaze_in_sensitive_bbox(100, 200, 100, 200, 400, 600) is True


def test_point_outside_bbox_no_radius():
    assert _is_gaze_in_sensitive_bbox(50, 300, 100, 200, 400, 600) is False


def test_radius_reaches_bbox():
    # gaze at (50, 300), bbox starts at x=100 — distance is 50px
    assert _is_gaze_in_sensitive_bbox(50, 300, 100, 200, 400, 600, radius=55) is True


def test_radius_does_not_reach_bbox():
    assert _is_gaze_in_sensitive_bbox(50, 300, 100, 200, 400, 600, radius=40) is False


def test_invalid_bbox_coords_swapped():
    # x2 < x1 is invalid
    assert _is_gaze_in_sensitive_bbox(200, 300, 400, 200, 100, 600) is False


def test_non_numeric_input():
    assert _is_gaze_in_sensitive_bbox("a", 300, 100, 200, 400, 600) is False


# ── _matched_label ─────────────────────────────────────────────────────────────

DETECTIONS = [
    {"label": "upper_torso", "bbox": [100, 200, 400, 600], "confidence": 0.9},
]


def test_matched_label_hit():
    assert _matched_label(DETECTIONS, 200, 300, 0) == "upper_torso"


def test_matched_label_miss_outside_bbox():
    assert _matched_label(DETECTIONS, 500, 300, 0) is None


def test_matched_label_low_confidence():
    dets = [{"label": "upper_torso", "bbox": [100, 200, 400, 600], "confidence": 0.1}]
    assert _matched_label(dets, 200, 300, 0) is None


def test_matched_label_wrong_label():
    dets = [{"label": "head", "bbox": [100, 200, 400, 600], "confidence": 0.9}]
    assert _matched_label(dets, 200, 300, 0) is None


def test_matched_label_empty_detections():
    assert _matched_label([], 200, 300, 0) is None


# ── _matched_confidence ────────────────────────────────────────────────────────

def test_matched_confidence_hit():
    assert _matched_confidence(DETECTIONS, 200, 300, 0) == pytest.approx(0.9)


def test_matched_confidence_miss():
    assert _matched_confidence(DETECTIONS, 500, 300, 0) == pytest.approx(0.0)


def test_matched_confidence_with_radius():
    # gaze at (50, 300), just outside bbox — radius=60 should reach it
    assert _matched_confidence(DETECTIONS, 50, 300, 60) == pytest.approx(0.9)
