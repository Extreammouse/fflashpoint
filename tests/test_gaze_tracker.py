import numpy as np
import pytest

from src.gaze_tracker import (
    L2CSFeatureExtractor,
    WebcamGazeTracker,
    _extract_l2cs_result_rows,
)


class _Face:
    def __init__(self, confidence):
        self.confidence = confidence


class _BatchResult:
    def __init__(self, yaw, pitch, face):
        self.yaw = yaw
        self.pitch = pitch
        self.face = face


class _DummyModel:
    screen_size = (2560, 1600)
    validation_error_px = 122.0
    training_error_px = 88.0


class _DummySmoother:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class _EmptyFacePipeline:
    def step(self, _frame):
        raise ValueError("need at least one array to stack")


def test_extract_l2cs_result_rows_handles_array_outputs():
    result = _BatchResult(
        yaw=np.array([-0.21, 0.08], dtype=np.float32),
        pitch=np.array([0.11, -0.04], dtype=np.float32),
        face=[_Face(0.91), _Face(0.73)],
    )

    rows = _extract_l2cs_result_rows(result)

    assert len(rows) == 2
    assert rows[0]["yaw"] == pytest.approx(-0.21)
    assert rows[0]["pitch"] == pytest.approx(0.11)
    assert rows[0]["confidence"] == pytest.approx(0.91)
    assert rows[1]["yaw"] == pytest.approx(0.08)
    assert rows[1]["pitch"] == pytest.approx(-0.04)
    assert rows[1]["confidence"] == pytest.approx(0.73)


def test_webcam_tracker_center_fallback_uses_screen_center_and_zero_confidence():
    tracker = WebcamGazeTracker.__new__(WebcamGazeTracker)
    tracker.model = _DummyModel()
    tracker.smoother = _DummySmoother()
    tracker.smoothing_alpha = 0.36
    tracker.smoothing_median_window = 3
    tracker.smoothing_max_step_px = 420

    estimate = tracker._center_fallback_estimate(
        status="face/eyes not found; center fallback",
        quality="no_face_center_fallback",
        debug={"status": "face/eyes not found"},
    )

    assert estimate.x == pytest.approx(1280.0)
    assert estimate.y == pytest.approx(800.0)
    assert estimate.confidence == 0.0
    assert estimate.source == "webcam_center_fallback"
    assert estimate.quality == "no_face_center_fallback"
    assert tracker.smoother.reset_called is True
    assert tracker.last_status == "face/eyes not found; center fallback"
    assert tracker.last_debug["fallback"] == "screen_center"


def test_l2cs_feature_extractor_returns_none_when_pipeline_has_no_faces():
    extractor = L2CSFeatureExtractor.__new__(L2CSFeatureExtractor)
    extractor._pipeline = _EmptyFacePipeline()

    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert extractor.extract(frame) is None
