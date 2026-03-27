# Flash Point — Detailed Sequence Diagram

## System Overview

Flash Point is a vision-based intervention system that detects when a user's gaze intersects with sensitive objects (e.g., heads) and triggers a full-screen white flash as an intervention.

---

## Complete Message Flow Sequence Diagram

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
│   Webcam    │      │   Video      │      │  Spark          │      │ Flash        │
│   (Gaze)    │      │  (Objects)   │      │  Analytics      │      │ Trigger      │
│             │      │              │      │  Processor      │      │              │
└──────┬──────┘      └──────┬───────┘      └────────┬────────┘      └──────┬───────┘
       │                    │                       │                      │
       │ Real-time gaze     │ Real-time frames      │                      │
       │ (MediaPipe)        │ (YOLOv8 + OpenCV)    │                      │
       │                    │                       │                      │
       ├────────────────────────────────────────────────────────────────────┤
       │ vision_node.py (Producer) - Runs @ ~24 FPS                        │
       └───────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   GAZE_EVENTS_TOPIC    │
                    │    (Kafka Topic)       │
                    │                        │
                    │ Message Schema:        │
                    │ {                      │
                    │  timestamp: float,     │
                    │  gaze_x: float,        │
                    │  gaze_y: float,        │
                    │  detected_objects: [   │
                    │   {label, bbox, conf}  │
                    │  ]                     │
                    │ }                      │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼────────────────────────────────────┐
                    │  spark_analytics.py (Processor)               │
                    │  Structured Streaming Consumer               │
                    │                                               │
                    │  1. Parse JSON messages                       │
                    │  2. For each gaze event:                      │
                    │     - Check if gaze_x, gaze_y inside          │
                    │       any sensitive object bbox               │
                    │     - Apply UDFs (gaze_udf.py):               │
                    │       * _is_gaze_in_sensitive_bbox()          │
                    │       * _matched_label()                      │
                    │       * _matched_confidence()                 │
                    │  3. Filter violations (True)                  │
                    │  4. Log to console                            │
                    │  5. Send violations downstream                │
                    └──────────┬────────────────────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │ INTERVENTION_TRIGGER_TOPIC      │
                    │    (Kafka Topic)                │
                    │                                 │
                    │ Violation Message Schema:       │
                    │ {                               │
                    │  timestamp: float,              │
                    │  violation: true,               │
                    │  gaze_x: float,                 │
                    │  gaze_y: float,                 │
                    │  matched_label: str,            │
                    │  confidence: float              │
                    │ }                               │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │ flash_trigger.py        │
                    │ (Consumer)              │
                    │                         │
                    │ 1. Listen to topic      │
                    │ 2. Receive violation    │
                    │ 3. Log event details    │
                    │ 4. Trigger flash        │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │ Full-screen White Flash │
                    │ (Tkinter + Threading)   │
                    │                         │
                    │ Duration: 200ms         │
                    │ (FLASH_DURATION_MS)     │
                    │                         │
                    │ Lifecycle:              │
                    │ 1. Fade in (alpha: 0→1) │
                    │ 2. Show white bg        │
                    │ 3. Sleep 200ms          │
                    │ 4. Fade out (alpha: 1→0)│
                    │ 5. Black bg             │
                    └─────────────────────────┘
```

---

## Detailed Timeline: Single Frame → Flash

### **Phase 1: Vision Node (Producer) — ~40ms per frame @ 24 FPS**

```
┌─ Frame acquisition ─────────────────────────────────────────┐
│  1. Video file (sample_video.mp4) or live stream read       │
│  2. Current gaze state {x: float, y: float} fetched (thread-safe) │
│  3. Frame → YOLOv8 detection (sensitive_labels: {"head"})   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Object extraction & filtering ─────────────────────────────┐
│  For each detected box:                                      │
│    • Extract label (e.g., "head")                           │
│    • Extract bbox [x1, y1, x2, y2] (pixels)                 │
│    • Extract confidence (0.0 – 1.0)                         │
│    • Keep only SENSITIVE_LABELS {"head"}                    │
│    • Round to 1 decimal precision                           │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Gaze capture (parallel thread) ────────────────────────────┐
│  Continuous in background:                                   │
│    • Webcam frame → MediaPipe FaceMesh (30 FPS)            │
│    • Extract eye landmarks (iris center)                    │
│    • Apply homography transform (if calibration exists)     │
│    • Update shared state {gaze_x, gaze_y} with lock        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Kafka message construction & publish ──────────────────────┐
│  Message: {                                                  │
│    timestamp: <unix epoch seconds>,                          │
│    gaze_x: <float rounded to 2 decimals>,                   │
│    gaze_y: <float rounded to 2 decimals>,                   │
│    detected_objects: [                                       │
│      {label: "head", bbox: [x1, y1, x2, y2], confidence: c} │
│    ]                                                         │
│  }                                                           │
│                                                              │
│  Serialization: JSON → UTF-8 bytes                           │
│  Topic: GAZE_EVENTS_TOPIC ("gaze_events")                   │
│  Broker: localhost:9092                                      │
└────────────────────────────────────────────────────────────┘
```

---

### **Phase 2: Spark Processor (Analytics) — ~10-30ms latency**

```
┌─ Kafka stream consumption ──────────────────────────────────┐
│  • Structured Streaming reads from "gaze_events"            │
│  • Auto-reconnect, at-least-once semantics                  │
│  • Checkpoint directory: /tmp/flash_point_checkpoint        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Message deserialization ──────────────────────────────────┐
│  UTF-8 bytes → JSON string                                   │
│  → Parsed against gaze_event_schema                          │
│  Schema validation: timestamp, gaze_x, gaze_y,              │
│                    detected_objects array                    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ UDF Application (gaze_udf.py) ────────────────────────────┐
│                                                              │
│  3 UDFs applied in parallel (Spark optimization):           │
│                                                              │
│  1. is_gaze_in_sensitive_bbox(gaze_x, gaze_y, objects)    │
│     └─ For each detected object:                            │
│        • Check if gaze point falls inside bbox              │
│        • (x1 ≤ gaze_x ≤ x2) AND (y1 ≤ gaze_y ≤ y2)        │
│        • Return True if ANY match found                      │
│        • Return False otherwise                             │
│     └─ Output: Boolean (violation flag)                      │
│                                                              │
│  2. matched_label(gaze_x, gaze_y, objects)                 │
│     └─ Find the first object with gaze inside bbox         │
│     └─ Return its label (e.g., "head") or ""               │
│     └─ Output: String                                        │
│                                                              │
│  3. matched_confidence(gaze_x, gaze_y, objects)            │
│     └─ Find the first object with gaze inside bbox         │
│     └─ Return its confidence score (0.0 – 1.0)             │
│     └─ Output: Float                                         │
│                                                              │
│  All three executed against same row, output                │
│  new columns: violation, matched_label, matched_confidence  │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Filter violations ────────────────────────────────────────┐
│  Keep only rows where violation == True                     │
│  (gaze is inside a sensitive object bbox)                   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Dual output:                                              ┐
│  a) Console output:                                         │
│     • All annotated rows logged with 5 columns:             │
│       timestamp | gaze_x | gaze_y | violation | matched_*  │
│     • Useful for debugging & monitoring                     │
│                                                              │
│  b) Kafka output (intervention_trigger topic):              │
│     • Only violation rows                                   │
│     • Reformat to clean schema                              │
│     • Publish for consumer                                  │
└────────────────────────────────────────────────────────────┘
```

---

### **Phase 3: Flash Trigger (Consumer) → UI Flash**

```
┌─ Kafka listener thread ────────────────────────────────────┐
│  • Continuous listen to "intervention_trigger" topic       │
│  • auto_offset_reset = "latest" (only new messages)        │
│  • Deserializer: UTF-8 bytes → JSON                        │
│  • For each message:                                        │
│    {                                                        │
│      timestamp, violation, gaze_x, gaze_y,                 │
│      matched_label, confidence                             │
│    }                                                        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Log & fire flash ────────────────────────────────────────┐
│  Console output:                                            │
│  "FLASH  label={matched_label}  conf={confidence:.2f}      │
│          gaze=({gaze_x:.1f}, {gaze_y:.1f})"                │
│                                                              │
│  Then: call fire_flash() → Thread-safe schedule via        │
│        root.after(0, _do_flash)                             │
└────────────────────────────────────────────────────────────┘
                            ↓
┌─ Tkinter flash animation (main thread) ───────────────────┐
│  _do_flash():                                               │
│    1. Set window bg to white                               │
│    2. Set alpha (opacity) to 1.0 (fully visible)           │
│    3. Schedule _end_flash() in FLASH_DURATION_MS (200ms)   │
│                                                              │
│  _end_flash() [fires after 200ms]:                         │
│    1. Set alpha to 0.0 (fully transparent)                 │
│    2. Set bg to black (for next flash if needed)           │
│    3. Window remains fullscreen & topmost                  │
└────────────────────────────────────────────────────────────┘
```

---

## **Configuration & Constants**

| Component | Parameter | Value | Purpose |
|-----------|-----------|-------|---------|
| **Kafka** | `KAFKA_BROKER` | `localhost:9092` | Broker address |
| | `GAZE_EVENTS_TOPIC` | `gaze_events` | Vision → Spark topic |
| | `TRIGGER_TOPIC` | `intervention_trigger` | Spark → Consumer topic |
| **Vision** | `TARGET_FPS` | `24` | Frame capture rate |
| | `VIDEO_PATH` | `data/sample_video.mp4` | Video source |
| | `MODEL_PATH` | `models/yolov8n.pt` | YOLOv8 weights |
| | `CALIB_PATH` | `calibration/homography.npy` | Gaze calibration matrix |
| | `SENSITIVE_LABELS` | `{"head"}` | Labels triggering intervention |
| **Flash** | `FLASH_DURATION_MS` | `200` | Flash visibility duration |

---

## **Data Schemas**

### Input: `gaze_events` Topic (Producer → Processor)

```json
{
  "timestamp": 1711533604.123,
  "gaze_x": 512.45,
  "gaze_y": 384.78,
  "detected_objects": [
    {
      "label": "head",
      "bbox": [100.5, 50.2, 250.8, 300.1],
      "confidence": 0.94
    }
  ]
}
```

### Output: `intervention_trigger` Topic (Processor → Consumer)

```json
{
  "timestamp": 1711533604.123,
  "violation": true,
  "gaze_x": 512.45,
  "gaze_y": 384.78,
  "matched_label": "head",
  "confidence": 0.94
}
```

---

## **Execution Commands**

### **1. Start Infrastructure**
```bash
docker-compose -f docker/docker-compose.yml up
```

### **2. Terminal 1: Vision Producer**
```bash
source .venv/bin/activate
python src/producers/vision_node.py
```

### **3. Terminal 2: Spark Processor**
```bash
source .venv/bin/activate
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export SPARK_HOME=$(brew --prefix)/opt/apache-spark/libexec
spark-submit \
  --py-files src/processors/gaze_udf.py \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \
  src/processors/spark_analytics.py
```

### **4. Terminal 3: Flash Consumer**
```bash
source .venv/bin/activate
python src/consumers/flash_trigger.py
```

---

## **System Behavior & Edge Cases**

| Scenario | Behavior |
|----------|----------|
| **Gaze outside all bboxes** | UDFs return `violation=False`, no message to trigger topic |
| **Multiple objects detected** | UDFs check all, return first match if any |
| **No objects detected** | `detected_objects = []`, UDFs return `violation=False` |
| **Gaze on bbox boundary** | Bbox check is inclusive (`≤`), so boundary gaze = violation |
| **Flash trigger queue overflow** | Kafka buffer/queue handles backpressure; no visual lag expected @ 24 FPS |
| **Kafka broker down** | Producers/consumers block and retry; Spark checkpoints preserve state |
| **Calibration file missing** | Homography matrix = `None`; gaze coordinates used raw (uncalibrated) |

---

## **Performance Characteristics**

| Stage | Latency | Throughput |
|-------|---------|-----------|
| Frame capture & YOLOv8 | ~30–40ms | 24 FPS |
| Gaze thread (MediaPipe) | ~30ms | 30 FPS (parallel) |
| Kafka produce latency | ~5ms | Async, non-blocking |
| Spark Structured Streaming | ~10–30ms | Micro-batch (500ms default) |
| Kafka consume + UDF eval | ~2–5ms | Per-message |
| Flash UI response | <1ms | Tkinter `after()` is immediate |
| **End-to-end (frame → flash)** | **~50–100ms** | ~10–20 Hz effective |

---

## **Key Insights**

1. **Vision Node (Producer)** runs at 24 FPS, capturing gaze + objects every ~42ms.
2. **Spark Processor** uses structured streaming with micro-batches; default batch window is 500ms, so multiple gaze events are grouped and processed together.
3. **Violation detection** is purely geometric (point-in-bbox check) using UDFs in `gaze_udf.py`.
4. **Flash Trigger** responds to violations with a 200ms white flash, immediately visible to the user.
5. **All communication** flows through Kafka, decoupling components and enabling horizontal scaling.
6. **Thread safety** is managed via locks in `vision_node.py` (gaze state) and Tkinter's `after()` in `flash_trigger.py`.
