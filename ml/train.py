from ultralytics import YOLO

model = YOLO("yolov8n.pt")

_ = model.train(
    data="ml/head_detection.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="ml/runs",
    name="head_v1",
    device="cpu",
)
