"""Kafka producer: reads webcam gaze + YOLO detections, publishes to gaze_events."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.geometry import find_gaze_hits
from src.gaze_tracker import WebcamGazeTracker
from src.detector import YoloScreenDetector
from src.screen_capture import PrimaryMonitorCapture
from src.utils.kafka_config import (
    BOOTSTRAP_SERVERS,
    GAZE_EVENTS_TOPIC,
    FLASH_DURATION_MS,
)


def _make_event(gaze_event, detections, hits, timestamp):
    return {
        "timestamp": round(timestamp, 6),
        "gaze_x": float(gaze_event["x"]) if gaze_event else None,
        "gaze_y": float(gaze_event["y"]) if gaze_event else None,
        "confidence": float(gaze_event.get("confidence", 0.0)) if gaze_event else 0.0,
        "detections": [
            {
                "label": d.get("label"),
                "bbox": list(d.get("bbox", [])),
                "confidence": float(d.get("confidence", 0.0)),
            }
            for d in detections
        ],
        "hits": [{"label": h.get("label"), "confidence": float(h.get("confidence", 0.0))} for h in hits],
    }


def run():
    try:
        from kafka import KafkaProducer
    except ImportError:
        raise SystemExit("kafka-python not installed. Run: pip install kafka-python")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
        linger_ms=5,
    )

    if not WebcamGazeTracker.calibration_exists(config.CALIBRATION_PATH):
        raise SystemExit(
            f"Calibration missing — run calibrate_gaze.py first. Expected {config.CALIBRATION_PATH}"
        )

    screen_capture = PrimaryMonitorCapture(monitor_index=config.SCREEN_MONITOR_INDEX)
    screen_detector = YoloScreenDetector(
        model_path=config.BODY_REGION_MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        image_size=config.YOLO_IMAGE_SIZE,
        device=config.YOLO_DEVICE,
        geometry_scale=config.DETECTION_GEOMETRY_SCALE,
    )
    gaze_tracker = WebcamGazeTracker(
        camera_index=config.CAMERA_INDEX,
        calibration_path=config.CALIBRATION_PATH,
        smoothing_alpha=config.L2CS_GAZE_SMOOTHING_ALPHA,
    )

    target_labels = {config.TARGET_LABEL}
    detection_interval = 1.0 / max(1, int(config.SCREEN_CAPTURE_FPS))
    last_detection_time = 0.0
    last_detections: list = []

    print(f"vision_node: publishing to Kafka topic '{GAZE_EVENTS_TOPIC}' on {BOOTSTRAP_SERVERS}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            now = time.time()
            estimate = gaze_tracker.read()
            gaze_event = estimate.as_event_point() if estimate else None

            if now - last_detection_time >= detection_interval:
                captured = screen_capture.read()
                last_detections = screen_detector.detect(captured.image_bgr)
                last_detection_time = now

            hits = find_gaze_hits(
                gaze_event,
                last_detections,
                target_labels,
                config.CONFIDENCE_THRESHOLD,
                config.GAZE_HIT_RADIUS_PX,
            )

            event = _make_event(gaze_event, last_detections, hits, now)
            producer.send(GAZE_EVENTS_TOPIC, event)
    except KeyboardInterrupt:
        print("\nvision_node: shutting down.")
    finally:
        gaze_tracker.close()
        screen_capture.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
