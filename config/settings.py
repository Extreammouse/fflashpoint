# Central config — import this everywhere instead of hardcoding

KAFKA_BROKER = "localhost:9092"
GAZE_EVENTS_TOPIC = "gaze_events"
TRIGGER_TOPIC = "intervention_trigger"
CHECKPOINT_DIR = "/tmp/flash_point_checkpoint"

SENSITIVE_LABELS = {"head"}
CONFIDENCE_THRESHOLD = 0.5
FLASH_DURATION_MS = 200

MODEL_PATH = "models/yolov8n.pt"
VIDEO_PATH = "data/sample_video.mp4"
CALIB_PATH = "calibration/homography.npy"
TARGET_FPS = 24
