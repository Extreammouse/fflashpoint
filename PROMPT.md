# PROMPT.md — Flash Point: Gaze-Triggered Aversion System

## Project Overview

**Flash Point** is a real-time big data pipeline that combines computer vision, eye-tracking, and stream processing to detect when a user's gaze lands on a machine-learning-defined "sensitive zone" in a video, and immediately triggers a full-screen flash as a deterrent. Built for a Big Data Analytics course using Apache Kafka and Apache Spark Structured Streaming.

---

## Architecture Summary

```
Vision Producer          Kafka Broker         Spark Structured        Kafka Topic           Flash Consumer
(vision_node.py)  ──►  gaze_events topic  ──►  Streaming           ──► intervention_trigger ──► (flash_trigger.py)
                                               (spark_analytics.py)                              └─► Full-screen Flash UI
```

### Component Breakdown

| Component | File | Role |
|---|---|---|
| Vision Producer | `vision_node.py` | Runs webcam + eye tracker + object detector; publishes `{gaze_x, gaze_y, detected_objects[]}` JSON to Kafka |
| Kafka Broker | `localhost:9092` | Message bus; topics: `gaze_events`, `intervention_trigger` |
| Spark Streaming | `spark_analytics.py` | Consumes `gaze_events`, applies UDF to check if gaze falls inside a sensitive bounding box |
| Flash Consumer | `flash_trigger.py` | Consumes `intervention_trigger` topic; launches full-screen white flash in a UI window |

---

## What You Are Building

### Goal
Show a video to a user. A computer vision model (trained on a specific target — e.g., **a human head**) detects the target region in each frame and defines a bounding box. A separate eye-tracking module records where on the screen the user is looking. If the gaze point falls **inside the bounding box of the target**, the system publishes a violation event, and the screen immediately **flashes bright white** as a deterrent/aversion signal.

- ✅ User looks at head → Flash triggers
- ❌ User looks at sky, floor, legs, background → No flash

---

## File & Module Specification

### 1. `vision_node.py` — Vision Producer

**Purpose:** Capture video frames, run object detection (trained on target class), run eye tracker, and stream gaze + bbox data to Kafka.

**Responsibilities:**
- Load a pre-trained or fine-tuned object detection model (e.g., YOLOv8 fine-tuned on target class such as "human head")
- Open video file or webcam feed frame-by-frame
- For each frame:
  - Run object detector → extract bounding boxes of target objects (class: head)
  - Run eye-tracker (e.g., via webcam + MediaPipe or Tobii SDK) → get `(gaze_x, gaze_y)` in screen/frame coordinates
  - Serialize to JSON: `{"timestamp": ..., "gaze_x": ..., "gaze_y": ..., "detected_objects": [{"label": "head", "bbox": [x1, y1, x2, y2], "confidence": ...}]}`
  - Publish to Kafka topic `gaze_events`

**Libraries:** `ultralytics` (YOLOv8), `opencv-python`, `mediapipe` (for gaze estimation), `kafka-python`

**Training Note:** You will need to fine-tune YOLOv8 (or similar) on a dataset of the target region (e.g., human heads). Use a dataset like HollywoodHeads or annotate custom images. Export the model weights (`.pt`) and load them in this module.

---

### 2. Kafka Broker Setup

**Topics to create:**
- `gaze_events` — raw gaze + bounding box events (high throughput, low latency)
- `intervention_trigger` — violation events (sparse, only on gaze hits)

**Configuration:**
- Broker: `localhost:9092`
- Partitions: 1 (for local dev), 3+ (for scale)
- Retention: short (e.g., 60 seconds) since data is real-time

**Setup commands:**
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topics
bin/kafka-topics.sh --create --topic gaze_events --bootstrap-server localhost:9092
bin/kafka-topics.sh --create --topic intervention_trigger --bootstrap-server localhost:9092
```

**This repository:** use `docker-compose -f docker/docker-compose.yml up -d` for local Kafka + Zookeeper, then create topics as above (or via your preferred method).

---

### 3. `spark_analytics.py` — Spark Structured Streaming Engine

**Purpose:** Consume `gaze_events` from Kafka, parse JSON, apply a UDF that checks if `(gaze_x, gaze_y)` falls inside any detected object's bounding box, and if so, write a violation record to `intervention_trigger`.

**Responsibilities:**
- Initialize SparkSession with Kafka connector
- Read stream from topic `gaze_events`
- Parse JSON schema: `{timestamp, gaze_x, gaze_y, detected_objects: [{label, bbox: [x1,y1,x2,y2], confidence}]}`
- Define and register a **PySpark UDF** `is_gaze_in_sensitive_bbox(gaze_x, gaze_y, detected_objects) → bool`:
  - Iterate over detected objects
  - For each object with label in `SENSITIVE_LABELS` (e.g., `["head"]`):
    - Check if `x1 <= gaze_x <= x2` AND `y1 <= gaze_y <= y2`
  - Return `True` if any match found
- Filter stream: rows where UDF returns `True` → write to `intervention_trigger` Kafka topic
- Also `writeStream → console` for debugging
- Rows where UDF returns `False` → dropped (no output to topics)

**Libraries:** `pyspark`, `pyspark-sql`, Kafka connector JAR (`spark-sql-kafka`)

**Spark Submit Command:**
```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.x.x \
  spark_analytics.py
```

**Note:** The current `src/processors/spark_analytics.py` uses a Python UDF and checks `label == 'sensitive'`; align labels/schema with `SENSITIVE_LABELS` and the producer JSON as you integrate.

---

### 4. `flash_trigger.py` — Flash Consumer

**Purpose:** Consume messages from `intervention_trigger` and trigger an immediate full-screen white flash on the display showing the video.

**Responsibilities:**
- Connect to Kafka, subscribe to `intervention_trigger`
- Poll for messages in a tight loop
- On message received:
  - Open or signal a full-screen Tkinter / PyGame / OpenCV window
  - Flash it white for ~200ms
  - Return to transparent/hidden state
- Should run as a background process alongside the video player

**Flash implementation options:**
- **Tkinter:** `root.configure(bg='white')`, `root.attributes('-fullscreen', True)`, hold 200ms, then hide
- **PyGame:** Fill screen with `(255, 255, 255)` for 2-3 frames
- **OpenCV overlay:** Blend white frame over current video display

**Libraries:** `kafka-python`, `tkinter` or `pygame`

---

## Machine Learning Component

### Model: Target Zone Detector (e.g., Human Head Detector)

**Goal:** Train or fine-tune an object detection model that can draw bounding boxes around the target region (e.g., human head) in video frames.

**Recommended approach:**
1. **Base Model:** YOLOv8n or YOLOv8s (small and fast for real-time)
2. **Dataset:** HollywoodHeads dataset, SCUT-HEAD, or annotate custom images using LabelImg/Roboflow
3. **Training:**
   ```bash
   yolo train model=yolov8n.pt data=head_detection.yaml epochs=50 imgsz=640
   ```
4. **Output:** `best.pt` weights — load in `vision_node.py`
5. **Inference speed target:** ≥ 24 FPS to stay real-time with video playback

**Training data format (YOLO):**
- Images: `.jpg` frames extracted from video
- Labels: `.txt` files with `class_id cx cy w h` (normalized 0–1)
- `head_detection.yaml`:
  ```yaml
  train: ./data/train
  val: ./data/val
  nc: 1
  names: ['head']
  ```

**This repository:** a base `models/yolov8n.pt` is present; fine-tuned weights can live under `ml/weights/` per the layout below.

---

## Eye Tracking

**Approach 1 — Webcam-based (free, lower accuracy):**
- Use MediaPipe Face Mesh to estimate gaze direction from iris position
- Project gaze onto screen coordinates using homography or calibration matrix
- Library: `mediapipe`, `opencv-python`

**Approach 2 — Tobii / Pupil Labs (high accuracy, requires hardware):**
- Use SDK to get raw `(gaze_x, gaze_y)` directly
- Plug into `vision_node.py` as a data source

**Calibration:** Run a 5-9 point calibration screen at startup to map eye vectors to screen pixel coordinates.

---

## Data Flow (Detailed)

```
[Video Frame] ──► [YOLOv8 Inference] ──► detected_objects: [{label: "head", bbox: [340,120,480,260]}]
[Webcam]      ──► [MediaPipe Gaze]   ──► gaze_x: 400, gaze_y: 190

vision_node.py publishes:
{
  "timestamp": 1711234567.123,
  "gaze_x": 400,
  "gaze_y": 190,
  "detected_objects": [
    {"label": "head", "bbox": [340, 120, 480, 260], "confidence": 0.91}
  ]
}

spark_analytics.py UDF evaluates:
  340 <= 400 <= 480 ✅
  120 <= 190 <= 260 ✅
  → VIOLATION

Publishes to intervention_trigger:
{
  "timestamp": 1711234567.123,
  "violation": true,
  "gaze_x": 400,
  "gaze_y": 190,
  "matched_label": "head"
}

flash_trigger.py receives message → triggers full-screen white flash
```

---

## Project Directory Structure

```
flash-point/
├── PROMPT.md
├── README.md
├── requirements.txt
│
├── ml/
│   ├── head_detection.yaml        # YOLO dataset config
│   ├── train.py                   # Fine-tuning script
│   ├── evaluate.py                # mAP / inference speed tests
│   └── weights/
│       └── best.pt                # Trained model weights
│
├── pipeline/                      # Reference layout (optional)
│   ├── vision_node.py
│   ├── spark_analytics.py
│   └── flash_trigger.py
│
├── calibration/
│   └── gaze_calibration.py        # Eye tracker calibration screen
│
├── config/
│   └── settings.py                # Kafka hosts, topic names, sensitive labels, bbox thresholds
│
└── tests/
    ├── test_udf.py                 # Unit tests for gaze-in-bbox UDF
    ├── test_producer.py            # Mock producer tests
    └── sample_events.json          # Sample gaze event payloads
```

### Layout in this repository

The runnable code is currently under `src/` (same roles as `pipeline/`):

| Role | Path |
|------|------|
| Vision Producer | `src/producers/vision_node.py` |
| Spark Streaming | `src/processors/spark_analytics.py` |
| Flash Consumer | `src/consumers/flash_trigger.py` |
| Shared Kafka helpers | `src/utils/kafka_config.py` |
| Docker Kafka / Zookeeper | `docker/docker-compose.yml` |
| Packaged YOLO weights (optional) | `models/yolov8n.pt` |

---

## `requirements.txt`

```
kafka-python>=2.0.2
pyspark>=3.4.0
ultralytics>=8.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
pygame>=2.5.0
numpy>=1.24.0
```

(`tkinter` is in the Python standard library on most installs; do not list it as a pip dependency.)

---

## Deliverables for Big Data Analytics Class

1. **Architecture diagram** — the sequence diagram in the screenshot (Kafka + Spark implemented; vision/flash are stubs acceptable)
2. **Working Kafka pipeline** — producer → topic → consumer with real JSON events
3. **Spark UDF** — `is_gaze_in_sensitive_bbox` with unit tests
4. **Trained ML model** — YOLOv8 fine-tuned on target class, with mAP score reported
5. **End-to-end demo video** — show gaze on target → flash trigger; gaze off target → no flash
6. **Report** — latency measurements (producer → flash), throughput (events/sec), model accuracy

---

## Implementation Priority Order

1. ✅ Set up Kafka locally, create topics, test with dummy producer/consumer
2. ✅ Build and test the Spark UDF in isolation (`test_udf.py`)
3. ✅ Implement `spark_analytics.py` with mock JSON input
4. ✅ Build `flash_trigger.py` — test flash UI independently
5. ✅ Collect/annotate head detection dataset, fine-tune YOLOv8
6. ✅ Implement webcam gaze estimation with MediaPipe
7. ✅ Integrate all into `vision_node.py`, run end-to-end
8. ✅ Calibrate + tune bbox threshold and flash timing

---

## Notes

- The Spark + Kafka components are the **core of the big data assignment** — these must be fully implemented
- `vision_node.py` and `flash_trigger.py` can be **stubs** for initial demos (send mock gaze data, print flash to console)
- For the ML model, if training time is limited, a **pretrained head detector** (e.g., from Roboflow Universe) can be used with citation
- Sensitive labels are configurable — swap `"head"` for any target class without changing pipeline logic
