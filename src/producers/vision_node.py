"""
Real vision producer.
  - Opens video file (or webcam) for the video shown to the user
  - Runs YOLOv8 to detect target objects (e.g., heads) per frame
  - Reads webcam for gaze estimation via MediaPipe
  - Publishes gaze + bbox events to gaze_events at ~24 Hz
"""

import json
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from kafka import KafkaProducer
from ultralytics import YOLO

KAFKA_BROKER = "localhost:9092"
TOPIC = "gaze_events"
VIDEO_PATH = "data/sample_video.mp4"
MODEL_PATH = "models/yolov8n.pt"
CALIB_PATH = "calibration/homography.npy"
SENSITIVE_LABELS = {"head"}
TARGET_FPS = 24


def gaze_loop(face_mesh, h_mat, gaze_lock, gaze_state):
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            ix = (lm[468].x + lm[473].x) / 2 * w
            iy = (lm[468].y + lm[473].y) / 2 * h_px
            if h_mat is not None:
                pt = cv2.perspectiveTransform(
                    np.array([[[ix, iy]]], dtype=np.float32), h_mat
                )
                ix, iy = float(pt[0][0][0]), float(pt[0][0][1])
            with gaze_lock:
                gaze_state["x"] = round(ix, 2)
                gaze_state["y"] = round(iy, 2)


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    model = YOLO(MODEL_PATH)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
    )
    h_mat = np.load(CALIB_PATH) if os.path.exists(CALIB_PATH) else None

    gaze_lock = threading.Lock()
    gaze_state = {"x": 0.0, "y": 0.0}

    gaze_thread = threading.Thread(
        target=gaze_loop,
        args=(face_mesh, h_mat, gaze_lock, gaze_state),
        daemon=True,
    )
    gaze_thread.start()

    cap = cv2.VideoCapture(VIDEO_PATH)
    delay = 1.0 / TARGET_FPS

    print(f"vision_node running — publishing to {TOPIC}")
    try:
        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            objects = []
            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                if label not in SENSITIVE_LABELS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                objects.append(
                    {
                        "label": label,
                        "bbox": [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2, 1),
                            round(y2, 1),
                        ],
                        "confidence": round(float(box.conf[0]), 3),
                    }
                )

            with gaze_lock:
                gx, gy = gaze_state["x"], gaze_state["y"]

            event = {
                "timestamp": time.time(),
                "gaze_x": gx,
                "gaze_y": gy,
                "detected_objects": objects,
            }
            producer.send(TOPIC, event)

            cv2.imshow("Flash Point — Video", frame)
            if cv2.waitKey(1) == ord("q"):
                break

            elapsed = time.time() - t0
            time.sleep(max(0, delay - elapsed))

    finally:
        cap.release()
        cv2.destroyAllWindows()
        producer.flush()
        print("vision_node stopped.")


if __name__ == "__main__":
    main()
