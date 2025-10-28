from ultralytics import YOLO

# Create and train YOLO model
model = YOLO("yolov8n.pt")  # lightweight model for faster training

model.train(
    data="EVST_DataModelMk3/data.yaml",
    epochs=50,
    imgsz=640,
    project="EVST_DataModelMk3/runs/detect",
    name="train"
)

# Export for Raspberry Pi
model.export(format="torchscript")
print("[SUCCESS] Training complete — TorchScript exported.")
