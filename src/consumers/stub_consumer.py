"""
Prints every message from intervention_trigger.
Use this to confirm Spark is publishing violations correctly.
"""

import json

from kafka import KafkaConsumer

KAFKA_BROKER = "localhost:9092"
TOPIC = "intervention_trigger"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="latest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

if __name__ == "__main__":
    print(f"Listening on {TOPIC} — Ctrl+C to stop")
    for msg in consumer:
        v = msg.value
        print(
            f"FLASH TRIGGER  "
            f"gaze=({v['gaze_x']:.1f}, {v['gaze_y']:.1f})  "
            f"label={v['matched_label']}  "
            f"conf={v['confidence']:.2f}  "
            f"ts={v['timestamp']:.3f}"
        )
