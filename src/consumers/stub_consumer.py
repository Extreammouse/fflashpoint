"""
Prints every message from intervention_trigger.
Use this to confirm Spark is publishing violations correctly.
"""

from kafka import KafkaConsumer
from src.utils.kafka_config import CONSUMER_CONFIG, TRIGGER_TOPIC

TOPIC = TRIGGER_TOPIC

consumer = KafkaConsumer(TOPIC, **CONSUMER_CONFIG)

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
