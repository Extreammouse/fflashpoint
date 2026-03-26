"""
Consumes intervention_trigger and triggers a full-screen white flash.
Uses tkinter (stdlib — no pip install needed).

Run alongside the video player in a separate terminal.
"""

import json
import threading
import tkinter as tk

from kafka import KafkaConsumer

KAFKA_BROKER = "localhost:9092"
TOPIC = "intervention_trigger"
FLASH_DURATION_MS = 200  # how long the flash stays white

root = tk.Tk()
root.title("Flash Point")
root.attributes("-fullscreen", True)
root.attributes("-topmost", True)
root.configure(bg="black")
root.wm_attributes("-alpha", 0.0)  # start fully transparent


def _do_flash():
    root.configure(bg="white")
    root.wm_attributes("-alpha", 1.0)
    root.after(FLASH_DURATION_MS, _end_flash)


def _end_flash():
    root.wm_attributes("-alpha", 0.0)
    root.configure(bg="black")


def fire_flash():
    """Called from the Kafka thread — schedule flash on the Tk main thread."""
    root.after(0, _do_flash)


def kafka_listener():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset="latest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
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
