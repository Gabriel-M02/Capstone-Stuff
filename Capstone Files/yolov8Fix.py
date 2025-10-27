from ultralytics import YOLO
model = YOLO("EVST_DataModelPrototypemk1/runs/detect/train/weights/best.pt")
model.export(format="torchscript", imgsz=640)
