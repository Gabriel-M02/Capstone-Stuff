from ultralytics import YOLO
model = YOLO("EVST_DataModelMk2/runs/detect/train/weights/best.pt")
model.export(format="torchscript", imgsz=320)
