from ultralytics import YOLO

model = YOLO("ml/runs/head_v1/weights/best.pt")
metrics = model.val(data="ml/head_detection.yaml")

print(f"mAP50:     {metrics.box.map50:.3f}")
print(f"mAP50-95:  {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")
