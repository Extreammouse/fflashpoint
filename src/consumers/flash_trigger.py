"""
Consumes intervention_trigger and triggers a full-screen white flash.
Uses tkinter (stdlib — no pip install needed).

Run alongside the video player in a separate terminal.
"""

import sys
import threading
import tkinter as tk
from pathlib import Path

from kafka import KafkaConsumer

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.kafka_config import CONSUMER_CONFIG, TRIGGER_TOPIC, FLASH_DURATION_MS

TOPIC = TRIGGER_TOPIC

root = tk.Tk()
root.title("Flash Point")
root.attributes("-fullscreen", True)
root.attributes("-topmost", True)
root.configure(bg="white")
root.withdraw()  # hide window completely at start


def _do_flash():
    root.configure(bg="white")
    root.deiconify()  # show window
    root.after(FLASH_DURATION_MS, _end_flash)


def _end_flash():
    root.withdraw()  # hide window again after flash


def fire_flash():
    """Called from the Kafka thread — schedule flash on the Tk main thread."""
    root.after(0, _do_flash)


def kafka_listener():
    consumer = KafkaConsumer(TOPIC, **CONSUMER_CONFIG)
    print(f"Flash trigger listening on {TOPIC}")
    for msg in consumer:
        v = msg.value
        print(
            f"FLASH  label={v.get('matched_label')}  "
            f"conf={v.get('confidence', 0):.2f}  "
            f"gaze=({v.get('gaze_x', 0):.1f}, {v.get('gaze_y', 0):.1f})"
        )
        fire_flash()


def main():
    thread = threading.Thread(target=kafka_listener, daemon=True)
    thread.start()
    print("Flash window ready — press Esc to quit")
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
