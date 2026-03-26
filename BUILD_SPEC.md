# Flash Point — Phased Build Spec

Everything runnable locally, phase by phase. **Source files in this repo** implement the steps below; use this document as the runbook and checklist.

---

## Prerequisites (install once)

```bash
# Python 3.10+
python3 --version

# Docker Desktop (for Kafka)
docker --version

# Java 11+ (for Spark)
java -version

# Python deps (use a venv on macOS Homebrew Python)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## PHASE 0 — Local Infrastructure (Kafka via Docker)

**Goal:** Kafka broker running locally.

**File:** [`docker/docker-compose.yml`](docker/docker-compose.yml) (Confluent Zookeeper + Kafka).

### Start and verify

```bash
cd docker
docker compose up -d
docker compose ps
```

Create topics (container name is usually `{project}-kafka-1`; if the project folder is `docker`, that is often `docker-kafka-1`):

```bash
docker exec -it docker-kafka-1 \
  kafka-topics --create --topic gaze_events \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

docker exec -it docker-kafka-1 \
  kafka-topics --create --topic intervention_trigger \
  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

docker exec -it docker-kafka-1 \
  kafka-topics --list --bootstrap-server localhost:9092
```

Expected topics: `gaze_events`, `intervention_trigger`.

### Teardown

```bash
docker compose down
docker compose down -v   # reset volumes / offsets
```

---

## PHASE 1 — Stub Producer + Stub Consumer (No ML, No Spark)

**Goal:** Prove Kafka plumbing. Fake producer sends schema-valid JSON; consumer prints violations.

| Role | File |
|------|------|
| Producer | [`src/producers/stub_producer.py`](src/producers/stub_producer.py) |
| Consumer | [`src/consumers/stub_consumer.py`](src/consumers/stub_consumer.py) |

```bash
# Terminal 1
python src/consumers/stub_consumer.py

# Terminal 2
python src/producers/stub_producer.py
```

Without Spark, `stub_consumer` will not print violations yet (only after Phase 2).

---

## PHASE 2 — Spark Structured Streaming (UDF + Filter)

**Goal:** [`src/processors/spark_analytics.py`](src/processors/spark_analytics.py) reads `gaze_events`, writes violations to `intervention_trigger`. Pure logic lives in [`src/processors/gaze_udf.py`](src/processors/gaze_udf.py) so **pytest does not need PySpark**.

```bash
# Terminal 1
python src/consumers/stub_consumer.py

# Terminal 2
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/processors/spark_analytics.py

# Terminal 3
python src/producers/stub_producer.py
```

**Expected:** Spark console shows batches; ~every 4th event is a hit; `stub_consumer` prints `FLASH TRIGGER` only for violations.

### Unit tests (no Spark, no Kafka)

```bash
pytest tests/test_udf.py -v
```

---

## PHASE 3 — Flash Trigger UI

**Goal:** Full-screen white flash on violation.

**File:** [`src/consumers/flash_trigger.py`](src/consumers/flash_trigger.py) (tkinter + Kafka thread).

```bash
# Terminal 1 — Spark (same as Phase 2)
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/processors/spark_analytics.py

# Terminal 2
python src/consumers/flash_trigger.py

# Terminal 3
python src/producers/stub_producer.py
```

---

## PHASE 4 — Real Vision Producer

**Goal:** Webcam gaze (MediaPipe) + YOLO on a video file → `gaze_events`.

| File | Purpose |
|------|---------|
| [`calibration/gaze_calibration.py`](calibration/gaze_calibration.py) | 5-point calibration → `calibration/homography.npy` |
| [`src/producers/vision_node.py`](src/producers/vision_node.py) | Video + YOLO + gaze thread → Kafka |

Place a video at **`data/sample_video.mp4`** (repo has `data/` with `.gitkeep` only). Default YOLO weights: **`models/yolov8n.pt`**. COCO uses class **`person`**, not **`head`**; for head class you need a fine-tuned model (Phase 5) and then point `MODEL_PATH` / `SENSITIVE_LABELS` accordingly.

```bash
python calibration/gaze_calibration.py
python src/producers/vision_node.py
```

---

## PHASE 5 — ML (YOLOv8 on target class)

| File | Purpose |
|------|---------|
| [`ml/head_detection.yaml`](ml/head_detection.yaml) | Dataset layout under `ml/data/` |
| [`ml/train.py`](ml/train.py) | Training entry |
| [`ml/evaluate.py`](ml/evaluate.py) | Validation metrics |

Populate `ml/data/images/train` and `ml/data/images/val` with YOLO-format labels, then:

```bash
python ml/train.py
python ml/evaluate.py
```

---

## Full local run (all phases)

```bash
cd docker && docker compose up -d && cd ..

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/processors/spark_analytics.py &

python src/consumers/flash_trigger.py &

# Real vision or stub:
python src/producers/vision_node.py
# or
python src/producers/stub_producer.py
```

### Smoke test (single Kafka event)

```bash
python3 - <<'EOF'
import json
import time
from kafka import KafkaProducer
p = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode(),
)
p.send(
    "gaze_events",
    {
        "timestamp": time.time(),
        "gaze_x": 400.0,
        "gaze_y": 200.0,
        "detected_objects": [
            {"label": "head", "bbox": [300, 100, 500, 300], "confidence": 0.95}
        ],
    },
)
p.flush()
print("Sent one HIT event")
EOF
```

---

## Config reference

[`config/settings.py`](config/settings.py) — central constants (Kafka, topics, thresholds, paths). Wire imports from here in producers/consumers when you refactor hardcoded strings.

---

## Phase checklist

```
[ ] Phase 0  — Docker Kafka up, both topics created
[ ] Phase 1  — stub_producer sends; Kafka reachable
[ ] Phase 2  — Spark reads gaze_events; violations on intervention_trigger; pytest tests/test_udf.py green
[ ] Phase 3  — flash_trigger visible full-screen flash
[ ] Phase 4a — homography.npy from calibration
[ ] Phase 4b — vision_node end-to-end (video + sample_video.mp4)
[ ] Phase 5  — YOLO trained; swap weights; report mAP
```
