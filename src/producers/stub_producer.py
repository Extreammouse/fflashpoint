"""
Sends synthetic gaze events to gaze_events at ~10 Hz.
Alternates between gaze ON the head bbox and gaze OFF it so you can
visually confirm which events should trigger a violation.
"""

import json
import random
import time

from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "gaze_events"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

HEAD_BBOX = [300, 100, 500, 300]  # x1, y1, x2, y2


def make_event(on_target: bool) -> dict:
    if on_target:
        gaze_x = random.uniform(HEAD_BBOX[0], HEAD_BBOX[2])  # inside bbox
        gaze_y = random.uniform(HEAD_BBOX[1], HEAD_BBOX[3])
    else:
        gaze_x = random.uniform(0, 280)  # left of bbox
        gaze_y = random.uniform(0, 280)

    return {
        "timestamp": time.time(),
        "gaze_x": round(gaze_x, 2),
        "gaze_y": round(gaze_y, 2),
        "detected_objects": [
            {
                "label": "head",
                "bbox": HEAD_BBOX,
                "confidence": round(random.uniform(0.75, 0.99), 3),
            }
        ],
    }


if __name__ == "__main__":
    print(f"Publishing to {TOPIC} on {KAFKA_BROKER} — Ctrl+C to stop")
    tick = 0
    try:
        while True:
            on_target = tick % 4 == 0  # every 4th event is ON target
            event = make_event(on_target)
            producer.send(TOPIC, event)
            label = "HIT " if on_target else "miss"
            print(f"[{label}] gaze=({event['gaze_x']:.1f}, {event['gaze_y']:.1f})")
            tick += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        producer.flush()
        print("Stopped.")
