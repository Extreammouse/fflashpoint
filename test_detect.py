"""Quick test: run YOLO detector on a local image and print what it finds."""

import sys
import cv2
import config
from src.detector import YoloScreenDetector

image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

img = cv2.imread(image_path)
if img is None:
    raise SystemExit(f"Could not read image: {image_path}")

detector = YoloScreenDetector(
    model_path=config.BODY_REGION_MODEL_PATH,
    confidence_threshold=0.3,   # lower threshold to catch more
    image_size=config.YOLO_IMAGE_SIZE,
    device=config.YOLO_DEVICE,
    geometry_scale=config.DETECTION_GEOMETRY_SCALE,
)

detections = detector.detect(img)
print(f"\nFound {len(detections)} detections:\n")
for d in detections:
    print(f"  label={d['label']}  conf={d['confidence']:.2f}  bbox={[round(x) for x in d['bbox']]}")

if not detections:
    print("  (nothing detected)")
