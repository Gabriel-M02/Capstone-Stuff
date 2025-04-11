import cv2
import numpy as np
import psutil
import time
from ultralytics import YOLO

# Load your custom-trained YOLOv8 model (for detecting both people and weapons)
model = YOLO("EVST_DataModelPrototypemk1/runs/detect/train/weights/best.pt")

# Battery HUD

def draw_battery_status(frame, x=540, y=10):
    battery = psutil.sensors_battery()
    if battery is None:
        return
    percent = battery.percent
    charging = battery.power_plugged
    width, height = 80, 30
    outline_color = (255, 255, 255)
    fill_color = (0, 255, 0) if percent > 20 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), outline_color, 2)
    cv2.rectangle(frame, (x + width, y + 8), (x + width + 6, y + height - 8), outline_color, -1)
    fill_width = int((percent / 100) * (width - 4))
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), fill_color, -1)
    text = f"{int(percent)}%{' ' if charging else ''}"
    cv2.putText(frame, text, (x - 60, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Night vision effect

def apply_night_vision_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_OCEAN)
    return colored

# Initialize capture
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 450))
    nv_frame = apply_night_vision_effect(frame.copy())

    # YOLO detection
    results = model(frame)[0]
    detections = results.boxes.xyxy.cpu().numpy() if results.boxes is not None and results.boxes.xyxy is not None else []
    classes = results.boxes.cls.cpu().numpy() if results.boxes is not None and results.boxes.cls is not None else []
    confidences = results.boxes.conf.cpu().numpy() if results.boxes is not None and results.boxes.conf is not None else []

    people = []
    guns = []

    for i in range(len(detections)):
        x1, y1, x2, y2 = map(int, detections[i])
        cls = int(classes[i])
        conf = float(confidences[i])

        if cls == 0 and conf >= 0.6:
            guns.append((x1, y1, x2, y2))
        elif cls == 1 and conf >= 0.6:
            people.append((x1, y1, x2, y2))

    # Classify each person as FRIENDLY or HOSTILE
    for person_box in people:
        px1, py1, px2, py2 = person_box
        label = "FRIENDLY"
        color = (0, 255, 0)

        for gun_box in guns:
            gx1, gy1, gx2, gy2 = gun_box
            if gx1 < px2 and gx2 > px1 and gy1 < py2 and gy2 > py1:
                label = "HOSTILE"
                color = (0, 0, 255)
                break

        cv2.rectangle(nv_frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(nv_frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Optional: draw gun boxes
    for gx1, gy1, gx2, gy2 in guns:
        cv2.rectangle(nv_frame, (gx1, gy1), (gx2, gy2), (255, 255, 0), 1)

    # HUD
    draw_battery_status(nv_frame)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(nv_frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(nv_frame, f"EVST NVG HUD MK 2 - {time.strftime('%H:%M:%S')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hybrid NVG Tracker - Multi Subject", nv_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
