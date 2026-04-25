"""Phase 1 app for the shipped L2CS + torso runtime."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import config
from src.alert import AlertController
from src.detector import YoloScreenDetector
from src.geometry import find_gaze_hits
from src.gaze_model import GazeEstimate
from src.gaze_tracker import WebcamGazeTracker
from src.overlay import (
    blank_preview,
    draw_alert_flash,
    draw_detection_boxes,
    draw_gaze_hit_circle,
    draw_gaze_trail,
    draw_soft_gaze_blob,
    screen_length_to_preview_length,
    draw_status_panel,
    screen_to_preview_point,
)
from src.screen_capture import PrimaryMonitorCapture
from src.screen_flash import (
    ScreenFlashOverlay,
    VALID_SCREEN_FLASH_MODES,
    resolve_screen_flash_style,
)

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised by manual setup.
    raise SystemExit(
        "Missing dependency: opencv-python. Run "
        ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    ) from exc


def main() -> None:
    args = parse_args()
    if args.alert and args.no_alert:
        raise SystemExit("Choose either --alert or --no-alert, not both.")
    if args.screen_flash and args.no_screen_flash:
        raise SystemExit("Choose either --screen-flash or --no-screen-flash, not both.")
    if args.screen_flash and args.no_alert:
        raise SystemExit("--screen-flash requires alerting; remove --no-alert.")

    target_labels = {config.TARGET_LABEL}
    screen_flash_enabled = (
        bool(config.SCREEN_FLASH_ENABLED or args.screen_flash)
        and not args.no_screen_flash
        and not args.no_alert
    )
    alert_enabled = bool(
        config.ALERT_ENABLED or args.alert or screen_flash_enabled
    ) and not args.no_alert
    screen_flash_mode, screen_flash_alpha, screen_flash_color = resolve_screen_flash_style(
        args.screen_flash_mode,
        overlay_alpha=config.SCREEN_FLASH_OVERLAY_ALPHA,
        overlay_color=config.SCREEN_FLASH_OVERLAY_COLOR,
        white_color=config.SCREEN_FLASH_WHITE_COLOR,
    )
    screen_capture, screen_detector, detection_mode, detection_status = setup_detection()
    screen_size = screen_capture.screen_size
    preview_size = make_preview_size(screen_size)
    preview_hit_radius = screen_length_to_preview_length(
        config.GAZE_HIT_RADIUS_PX,
        preview_size,
        screen_size,
    )
    alert_controller = AlertController(
        enabled=alert_enabled,
        dwell_ms=config.GAZE_DWELL_MS,
        duration_ms=config.ALERT_DURATION_MS,
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
    )
    screen_flash = ScreenFlashOverlay(
        enabled=screen_flash_enabled,
        duration_ms=config.ALERT_DURATION_MS,
        mode=screen_flash_mode,
        alpha=screen_flash_alpha,
        color=screen_flash_color,
        bounds=screen_capture.monitor,
    )

    logging_enabled = config.METADATA_LOGGING_ENABLED and not args.no_log
    event_log_path = Path(config.EVENT_LOG_PATH)
    if logging_enabled:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_enabled = config.DIAGNOSTIC_LOGGING_ENABLED and not args.no_log
    diagnostic_log_path = Path(config.DIAGNOSTIC_LOG_PATH)
    if diagnostics_enabled:
        diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_exists = WebcamGazeTracker.calibration_exists(config.CALIBRATION_PATH)
    if not calibration_exists:
        raise SystemExit(
            f"Calibration missing: run calibrate_gaze.py first. Expected {config.CALIBRATION_PATH}"
        )

    webcam_tracker = WebcamGazeTracker(
        camera_index=config.CAMERA_INDEX,
        calibration_path=config.CALIBRATION_PATH,
        smoothing_alpha=config.L2CS_GAZE_SMOOTHING_ALPHA,
    )
    gaze_mode = "webcam_calibrated"
    tracker_status = "webcam calibrated gaze active"
    calibration_summary = webcam_tracker.calibration_summary()

    cv2.namedWindow(config.PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.PREVIEW_WINDOW_NAME, preview_size[0], preview_size[1])

    trail: deque[tuple[int, int]] = deque(maxlen=config.GAZE_TRAIL_LENGTH)
    frame_count = 0
    fps = 0.0
    fps_t0 = time.perf_counter()
    last_log_time = 0.0
    last_diagnostic_log_time = 0.0
    last_detection_time = 0.0
    detection_interval = 1.0 / max(1, int(config.SCREEN_CAPTURE_FPS))
    last_screen_frame = None
    last_detections: list[dict[str, Any]] = []

    print("Phase 1 app running. Press q or Esc to quit.")
    print(f"Using webcam gaze from {config.CALIBRATION_PATH}")
    print(f"Detection mode: {detection_mode} ({detection_status})")
    print(f"Screen coordinates: {screen_size[0]}x{screen_size[1]} primary monitor")
    print("No screenshots, screen frames, videos, or webcam recordings are written.")

    try:
        while True:
            now = time.time()
            frame_count += 1
            fps_elapsed = time.perf_counter() - fps_t0
            if fps_elapsed >= 0.5:
                fps = frame_count / fps_elapsed
                frame_count = 0
                fps_t0 = time.perf_counter()

            if now - last_detection_time >= detection_interval or last_screen_frame is None:
                captured = screen_capture.read()
                last_screen_frame = captured.image_bgr
                last_detections = screen_detector.detect(captured.image_bgr)
                detection_status = (
                    f"live_yolo ok objects={len(last_detections)} "
                    f"device={screen_detector.device}"
                )
                last_detection_time = now
            detections = list(last_detections)

            gaze_event = current_gaze_event(webcam_tracker=webcam_tracker)
            tracker_status = webcam_tracker.last_status

            hit_gated = False
            hit_gate_reason = None
            hit_test_gaze = gaze_event if gaze_confidence_ok(gaze_event) else None
            if hit_test_gaze is None and gaze_event is not None:
                hit_gated = True
                hit_gate_reason = "low_gaze_confidence"
            hits = find_gaze_hits(
                hit_test_gaze,
                detections,
                target_labels,
                config.CONFIDENCE_THRESHOLD,
                config.GAZE_HIT_RADIUS_PX,
            )
            alert_state = alert_controller.update(time.monotonic(), hits).as_event()
            alert_state["screen_flash"] = screen_flash.as_metadata()
            if alert_state.get("triggered"):
                screen_flash.trigger(alert_state.get("matched_label"))
            visible_detections = visible_detections_for_preview(detections, target_labels)

            frame = make_preview_frame(
                last_screen_frame,
                preview_size,
                detection_mode,
            )
            preview_gaze = None
            if gaze_event is not None:
                preview_gaze = screen_to_preview_point(
                    (gaze_event["x"], gaze_event["y"]), preview_size, screen_size
                )
                trail.append(preview_gaze)
            else:
                trail.clear()

            draw_detection_boxes(
                frame,
                visible_detections,
                hits,
                preview_size,
                screen_size,
                target_labels,
            )
            draw_gaze_trail(frame, list(trail))
            draw_soft_gaze_blob(
                frame,
                preview_gaze,
                radius=max(config.GAZE_BLOB_RADIUS, preview_hit_radius),
                alpha=config.GAZE_BLOB_ALPHA,
            )
            draw_gaze_hit_circle(
                frame,
                preview_gaze,
                preview_hit_radius,
            )
            draw_alert_flash(frame, alert_state)
            draw_status_panel(
                frame,
                status_lines(
                    gaze_event,
                    hits,
                    fps,
                    screen_size,
                    preview_size,
                    gaze_mode,
                    tracker_status,
                    calibration_summary,
                    hit_gated,
                    detection_mode,
                    detection_status,
                    alert_state,
                ),
            )

            if logging_enabled and (now - last_log_time >= config.EVENT_LOG_INTERVAL_SECONDS):
                append_event(
                    event_log_path,
                    now,
                    gaze_event,
                    visible_detections,
                    hits,
                    calibration_summary,
                    detection_mode,
                    detection_status,
                    alert_state,
                )
                last_log_time = now
            if diagnostics_enabled and (
                now - last_diagnostic_log_time >= config.DIAGNOSTIC_LOG_INTERVAL_SECONDS
            ):
                append_diagnostic_event(
                    diagnostic_log_path,
                    timestamp=now,
                    gaze_mode=gaze_mode,
                    gaze_point=gaze_event,
                    tracker_status=tracker_status,
                    tracker_debug=webcam_tracker.last_debug if webcam_tracker is not None else {},
                    calibration_summary=calibration_summary,
                    hit_gated=hit_gated,
                    hit_gate_reason=hit_gate_reason,
                    detected_count=len(visible_detections),
                    matched_count=len(hits),
                    detection_mode=detection_mode,
                    detection_status=detection_status,
                    alert_state=alert_state,
                )
                last_diagnostic_log_time = now

            cv2.imshow(config.PREVIEW_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        webcam_tracker.close()
        screen_capture.close()
        screen_flash.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live gaze + torso trigger prototype.")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use the calibrated webcam runtime path. This is the shipped default behavior.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable metadata-only JSONL event logging for this run.",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Enable the controlled visual alert for sustained target hits.",
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Disable the controlled visual alert even if config enables it.",
    )
    parser.add_argument(
        "--screen-flash",
        action="store_true",
        help="Enable a topmost full-screen visual flash when the alert triggers.",
    )
    parser.add_argument(
        "--screen-flash-mode",
        choices=sorted(VALID_SCREEN_FLASH_MODES),
        default=config.SCREEN_FLASH_MODE,
        help="Full-screen flash style: overlay is semi-transparent; white is opaque white.",
    )
    parser.add_argument(
        "--no-screen-flash",
        action="store_true",
        help="Disable the full-screen flash even if config enables it.",
    )
    return parser.parse_args()


def setup_detection() -> tuple[PrimaryMonitorCapture, YoloScreenDetector, str, str]:
    screen_capture = PrimaryMonitorCapture(monitor_index=config.SCREEN_MONITOR_INDEX)
    screen_detector = YoloScreenDetector(
        model_path=config.BODY_REGION_MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        image_size=config.YOLO_IMAGE_SIZE,
        device=config.YOLO_DEVICE,
        geometry_scale=config.DETECTION_GEOMETRY_SCALE,
    )
    return (
        screen_capture,
        screen_detector,
        "live_yolo",
        (
            f"live screen capture active model={Path(config.BODY_REGION_MODEL_PATH).name} "
            f"device={screen_detector.device}"
        ),
    )


def visible_detections_for_preview(
    detections: list[dict[str, Any]],
    target_labels: set[str],
) -> list[dict[str, Any]]:
    return [
        detection
        for detection in detections
        if isinstance(detection, dict) and detection.get("label") in target_labels
    ]


def make_preview_size(screen_size: tuple[int, int]) -> tuple[int, int]:
    """Fit a primary-monitor preview inside configured max dimensions."""
    screen_width, screen_height = screen_size
    scale = min(
        config.PREVIEW_MAX_WIDTH / max(1, screen_width),
        config.PREVIEW_MAX_HEIGHT / max(1, screen_height),
        1.0,
    )
    return max(320, int(screen_width * scale)), max(240, int(screen_height * scale))


def make_preview_frame(
    screen_frame: Any,
    preview_size: tuple[int, int],
    detection_mode: str,
) -> Any:
    """Return the preview canvas.

    The captured screen is not shown by default because the OpenCV window is
    part of the captured desktop and creates a recursive mirror.
    """
    return blank_preview(preview_size)


def current_gaze_event(
    webcam_tracker: WebcamGazeTracker,
) -> dict[str, Any] | None:
    estimate = webcam_tracker.read()
    if estimate is None:
        return None
    return format_gaze_estimate(estimate)


def append_event(
    event_log_path: Path,
    timestamp: float,
    gaze_point: dict[str, Any] | None,
    detections: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    calibration_summary: dict[str, Any],
    detection_mode: str,
    detection_status: str,
    alert_state: dict[str, Any],
) -> None:
    """Append one metadata-only event. No image or frame data is stored."""
    event = {
        "timestamp": round(timestamp, 6),
        "gaze_point": gaze_point,
        "detected_objects": compact_detections_for_log(detections),
        "matched_objects": compact_detections_for_log(hits),
        "calibration": calibration_summary,
        "detector": {"mode": detection_mode, "status": detection_status},
        "trigger": alert_state,
    }
    with event_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def compact_detections_for_log(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep event logs metadata-only and avoid dumping segmentation polygons."""
    compact: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        item = {
            "label": detection.get("label"),
            "bbox": detection.get("bbox"),
            "confidence": detection.get("confidence"),
        }
        if detection.get("has_mask"):
            item["has_mask"] = True
            item["mask_point_count"] = detection.get("mask_point_count")
        if "geometry_scale" in detection:
            item["geometry_scale"] = detection.get("geometry_scale")
        if "hit_region" in detection:
            item["hit_region"] = detection.get("hit_region")
        if "gaze_radius_px" in detection:
            item["gaze_radius_px"] = detection.get("gaze_radius_px")
        compact.append(item)
    return compact


def append_diagnostic_event(
    diagnostic_log_path: Path,
    timestamp: float,
    gaze_mode: str,
    gaze_point: dict[str, Any] | None,
    tracker_status: str,
    tracker_debug: dict[str, Any],
    calibration_summary: dict[str, Any],
    hit_gated: bool,
    hit_gate_reason: str | None,
    detected_count: int,
    matched_count: int,
    detection_mode: str,
    detection_status: str,
    alert_state: dict[str, Any],
) -> None:
    event = {
        "timestamp": round(timestamp, 6),
        "gaze_mode": gaze_mode,
        "gaze_point": gaze_point,
        "tracker_status": tracker_status,
        "tracker_debug": tracker_debug,
        "calibration": calibration_summary,
        "detector": {"mode": detection_mode, "status": detection_status},
        "trigger": alert_state,
        "hit_gate": {"gated": hit_gated, "reason": hit_gate_reason},
        "counts": {"detected": detected_count, "matched": matched_count},
    }
    with diagnostic_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def format_gaze_estimate(estimate: GazeEstimate) -> dict[str, Any]:
    return estimate.as_event_point()


def gaze_confidence_ok(gaze_point: dict[str, Any] | None) -> bool:
    if gaze_point is None:
        return False
    try:
        return float(gaze_point.get("confidence", 0.0)) >= config.GAZE_MIN_CONFIDENCE_FOR_HIT
    except (TypeError, ValueError):
        return False


def status_lines(
    gaze_point: dict[str, Any] | None,
    hits: list[dict[str, Any]],
    fps: float,
    screen_size: tuple[int, int],
    preview_size: tuple[int, int],
    gaze_mode: str,
    tracker_status: str,
    calibration_summary: dict[str, Any],
    hit_gated: bool,
    detection_mode: str,
    detection_status: str,
    alert_state: dict[str, Any],
) -> list[str]:
    if gaze_point is None:
        gaze_text = "gaze: unavailable"
    else:
        gaze_text = (
            f"gaze: ({float(gaze_point['x']):.0f}, {float(gaze_point['y']):.0f}) px "
            f"conf={float(gaze_point.get('confidence', 0.0)):.2f}"
        )

    hit_text = "hit: none"
    if hit_gated:
        hit_text = "hit: gated low gaze confidence"
    if hits:
        hit = hits[0]
        hit_region = hit.get("hit_region", "bbox")
        hit_text = (
            f"hit: {hit.get('label')} {hit_region} "
            f"conf={float(hit.get('confidence', 0.0)):.2f}"
        )

    error = calibration_summary.get("average_error_px")
    error_text = "n/a" if error is None else f"{float(error):.0f}px"

    return [
        f"Phase 1 | fps={fps:.1f} | gaze_mode={gaze_mode}",
        f"screen={screen_size[0]}x{screen_size[1]} preview={preview_size[0]}x{preview_size[1]}",
        f"detector={detection_mode} | {detection_status}",
        f"{gaze_text} | radius={config.GAZE_HIT_RADIUS_PX}px | targets=['{config.TARGET_LABEL}'] | {hit_text}",
        f"alert={format_alert_status(alert_state)} | calibration_error={error_text} | quit: q or Esc",
        f"tracker={tracker_status}",
    ]


def format_alert_status(alert_state: dict[str, Any]) -> str:
    if not alert_state.get("enabled"):
        return "disabled"
    if alert_state.get("active"):
        return f"active:{alert_state.get('active_label') or 'target'}"
    cooldown = float(alert_state.get("cooldown_remaining_s") or 0.0)
    if cooldown > 0.0:
        return f"cooldown:{cooldown:.1f}s"
    dwell = int(alert_state.get("dwell_progress_ms") or 0)
    dwell_required = int(alert_state.get("dwell_ms") or 0)
    if dwell > 0 and dwell_required > 0:
        return f"dwell:{dwell}/{dwell_required}ms"
    flash = alert_state.get("screen_flash")
    if isinstance(flash, dict) and flash.get("enabled"):
        return f"armed flash:{flash.get('mode')}:{flash.get('status')}"
    return "armed"
if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
